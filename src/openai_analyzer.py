#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI API integration for analyzing job descriptions."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence

from openai import OpenAI

from .config import (
    MAX_JOB_DESC_LENGTH,
    OPENAI_BATCH_SIZE,
    OPENAI_MAX_PARALLEL_REQUESTS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    RATE_LIMIT_DELAY,
)
from .models import JobAnalysisResult
from .prompts import job_analysis_batch_prompt, job_analysis_instructions

JOB_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "has_ai_skill": {"type": "boolean"},
        "ai_skills_mentioned": {"type": "array", "items": {"type": "string"}},
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": (
                "Confidence for both has_ai_skill and ai_skills_mentioned answers"
            ),
        },
    },
    "required": ["id", "has_ai_skill", "ai_skills_mentioned", "confidence"],
    "additionalProperties": False,
}

BATCH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {"type": "array", "items": JOB_RESULT_SCHEMA},
    },
    "required": ["results"],
    "additionalProperties": False,
}


class OpenAIJobAnalyzer:
    """Encapsulates the OpenAI client and response parsing logic."""

    def __init__(
        self,
        *,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        delay_seconds: float = RATE_LIMIT_DELAY,
        batch_size: int = OPENAI_BATCH_SIZE,
        max_concurrent_requests: int = OPENAI_MAX_PARALLEL_REQUESTS,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.delay_seconds = delay_seconds
        self.batch_size = max(1, batch_size)
        self.max_concurrent_requests = max(1, max_concurrent_requests)

    def analyze_text(self, job_desc_text: Optional[str]) -> JobAnalysisResult:
        """Run the LLM prompt for a single job description."""
        return self.analyze_texts([job_desc_text])[0]

    def analyze_texts(
        self, job_desc_texts: Sequence[Optional[str]]
    ) -> list[JobAnalysisResult]:
        """Analyze multiple job descriptions using batched OpenAI requests."""
        if not job_desc_texts:
            return []

        normalized_texts = [self._prepare_text(text) for text in job_desc_texts]
        results: list[JobAnalysisResult] = [
            JobAnalysisResult() for _ in normalized_texts
        ]

        pending_batches: list[tuple[list[tuple[str, str]], dict[str, int]]] = []
        batch: list[tuple[str, str]] = []
        index_lookup: dict[str, int] = {}
        for idx, text in enumerate(normalized_texts):
            if not text:
                continue
            job_id = f"job_{idx}"
            batch.append((job_id, text))
            index_lookup[job_id] = idx
            if len(batch) >= self.batch_size:
                pending_batches.append((batch, index_lookup))
                batch = []
                index_lookup = {}

        if batch:
            pending_batches.append((batch, index_lookup))

        self._process_batches(pending_batches, results)

        return results

    def _prepare_text(self, job_desc_text: Optional[str]) -> Optional[str]:
        if job_desc_text is None:
            return None

        text = str(job_desc_text).strip()
        if not text or text.lower() == "nan":
            return None

        if len(text) > MAX_JOB_DESC_LENGTH:
            return text[:MAX_JOB_DESC_LENGTH]
        return text

    def _process_batches(
        self,
        pending_batches: list[tuple[list[tuple[str, str]], dict[str, int]]],
        results: list[JobAnalysisResult],
    ) -> None:
        if not pending_batches:
            return

        max_workers = min(self.max_concurrent_requests, len(pending_batches))
        if max_workers <= 1:
            for batch, lookup in pending_batches:
                self._dispatch_batch(batch, lookup, results)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._dispatch_batch, batch, lookup, results)
                for batch, lookup in pending_batches
            ]
            for future in futures:
                future.result()

    def _dispatch_batch(
        self,
        batch: list[tuple[str, str]],
        index_lookup: dict[str, int],
        results: list[JobAnalysisResult],
    ) -> None:
        response_payload = self._call_openai_batch(batch)
        parsed_entries = self._parse_batch_response(response_payload)
        for entry in parsed_entries:
            job_id = entry.get("id")
            if job_id not in index_lookup:
                continue
            results[index_lookup[job_id]] = self._to_result(entry)

        time.sleep(self.delay_seconds)

    def _call_openai_batch(self, batch_items: list[tuple[str, str]]) -> str:
        system_prompt = (
            "You are an expert at analyzing job descriptions for AI and "
            "machine learning skills. Always respond with valid JSON only.\n\n"
            f"{job_analysis_instructions()}"
        )
        prompt = job_analysis_batch_prompt(batch_items)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            temperature=self.temperature,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "job_analysis_batch_result",
                    "schema": BATCH_RESPONSE_SCHEMA,
                }
            },
        )
        return self._extract_response_text(response)

    @staticmethod
    def _parse_batch_response(response_text: str) -> list[dict[str, Any]]:
        if not response_text:
            return []
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as error:
            print(f"Warning: Failed to parse OpenAI JSON response: {error}")
            return []

        results = parsed.get("results", [])
        if not isinstance(results, list):
            return []
        normalized: list[dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            normalized.append(entry)
        return normalized

    @staticmethod
    def _to_result(payload: dict[str, Any]) -> JobAnalysisResult:
        skills = payload.get("ai_skills_mentioned", [])
        if not isinstance(skills, list):
            skills = []
        normalized_skills = [skill for skill in skills if isinstance(skill, str)]
        return JobAnalysisResult(
            has_ai_skill=bool(payload.get("has_ai_skill", False)),
            ai_skills_mentioned=normalized_skills,
            confidence=_coerce_confidence(payload.get("confidence")),
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Handle extraction for Responses API while staying backward compatible."""
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        output_blocks = getattr(response, "output", None) or []
        for block in output_blocks:
            contents = getattr(block, "content", None) or []
            for item in contents:
                text_value = getattr(item, "text", None)
                if text_value:
                    return text_value.strip()

        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()

        return ""


def _coerce_confidence(value: Any) -> float:
    """Convert the model confidence to a bounded float."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
