#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI API integration for analyzing job descriptions."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from openai import OpenAI
from pydantic import ValidationError

from .config import (
    MAX_JOB_DESC_LENGTH,
    OPENAI_BATCH_SIZE,
    OPENAI_MAX_PARALLEL_REQUESTS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    RATE_LIMIT_DELAY,
)
from .models import BatchAnalysisResponse, JobAnalysisResult, JobAnalysisResultWithId
from .prompts import job_analysis_batch_prompt, job_analysis_instructions


def _prepare_schema_for_openai(schema: dict) -> dict:
    """
    Prepare Pydantic-generated JSON schema for OpenAI's Responses API.
    
    OpenAI's Responses API requires:
    1. additionalProperties: False for strict validation
    2. All properties must be in the 'required' array (even if they have defaults)
    """
    schema = schema.copy()
    
    def prepare_object_schema(obj: dict) -> None:
        """Recursively prepare object schemas for OpenAI's strict requirements."""
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                # Set additionalProperties: False
                obj["additionalProperties"] = False
                
                # Ensure all properties are in the required array
                properties = obj.get("properties", {})
                if properties:
                    # OpenAI requires ALL properties to be in the required array
                    obj["required"] = sorted(list(properties.keys()))
            
            # Recursively process nested schemas
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                prepare_object_schema(item)
                    else:
                        prepare_object_schema(value)
    
    # Process the main schema
    prepare_object_schema(schema)
    
    # Also process any schema definitions ($defs) that Pydantic may have generated
    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            prepare_object_schema(def_schema)
    
    return schema


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

    def analyze_text(
        self,
        job_desc_text: Optional[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> JobAnalysisResult:
        """Run the LLM prompt for a single job description."""
        return self.analyze_texts(
            [job_desc_text], progress_callback=progress_callback
        )[0]

    def analyze_texts(
        self,
        job_desc_texts: Sequence[Optional[str]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[JobAnalysisResult]:
        """Analyze multiple job descriptions using batched OpenAI requests."""
        if not job_desc_texts:
            return []

        normalized_texts = [self._prepare_text(text) for text in job_desc_texts]
        results: list[JobAnalysisResult] = [
            JobAnalysisResult() for _ in normalized_texts
        ]
        total_trackable = sum(1 for text in normalized_texts if text is not None)

        if progress_callback and total_trackable:
            progress_callback(0, total_trackable)
        processed_counter = 0
        progress_lock = threading.Lock()

        def report_progress(increment: int) -> None:
            nonlocal processed_counter
            if (
                not progress_callback
                or total_trackable == 0
                or increment <= 0
            ):
                return
            with progress_lock:
                processed_counter = min(
                    total_trackable, processed_counter + increment
                )
                progress_callback(processed_counter, total_trackable)

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

        self._process_batches(
            pending_batches, results, progress_reporter=report_progress
        )

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
        *,
        progress_reporter: Callable[[int], None] | None = None,
    ) -> None:
        if not pending_batches:
            return

        max_workers = min(self.max_concurrent_requests, len(pending_batches))
        if max_workers <= 1:
            for batch, lookup in pending_batches:
                self._dispatch_batch(
                    batch, lookup, results, progress_reporter=progress_reporter
                )
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._dispatch_batch,
                    batch,
                    lookup,
                    results,
                    progress_reporter=progress_reporter,
                )
                for batch, lookup in pending_batches
            ]
            for future in futures:
                future.result()

    def _dispatch_batch(
        self,
        batch: list[tuple[str, str]],
        index_lookup: dict[str, int],
        results: list[JobAnalysisResult],
        *,
        progress_reporter: Callable[[int], None] | None = None,
    ) -> None:
        response_payload = self._call_openai_batch(batch)
        parsed_entries = self._parse_batch_response(response_payload)
        for entry in parsed_entries:
            job_id = entry.id
            if job_id not in index_lookup:
                continue
            results[index_lookup[job_id]] = self._to_result(entry)
        if progress_reporter:
            progress_reporter(len(index_lookup))

        time.sleep(self.delay_seconds)

    def _call_openai_batch(self, batch_items: list[tuple[str, str]]) -> str:
        system_prompt = (
            "You are an expert at analyzing job descriptions for AI and "
            "machine learning skills. Always respond with valid JSON only.\n\n"
            f"{job_analysis_instructions()}"
        )
        prompt = job_analysis_batch_prompt(batch_items)
        
        # Generate JSON schema from Pydantic model and prepare for OpenAI
        raw_schema = BatchAnalysisResponse.model_json_schema()
        schema = _prepare_schema_for_openai(raw_schema)
        
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
                    "schema": schema,
                }
            },
        )
        return self._extract_response_text(response)

    @staticmethod
    def _parse_batch_response(response_text: str) -> list[JobAnalysisResultWithId]:
        """Parse and validate batch response using Pydantic models."""
        if not response_text:
            return []
        
        try:
            # Use Pydantic's model_validate_json for direct JSON string parsing and validation
            batch_response = BatchAnalysisResponse.model_validate_json(response_text)
            return batch_response.results
        except ValidationError as error:
            # Provide detailed validation error information
            error_details = []
            for err in error.errors():
                field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
                error_msg = err.get("msg", "Unknown error")
                error_type = err.get("type", "unknown")
                error_details.append(
                    f"  Field '{field_path}': {error_msg} (type: {error_type})"
                )
            
            print(f"Warning: Failed to validate OpenAI response with Pydantic:")
            print(f"  Error summary: {error}")
            if error_details:
                print("  Field-level errors:")
                for detail in error_details:
                    print(detail)
            print(f"  Response text (first 500 chars): {response_text[:500]}...")
            return []
        except Exception as error:
            print(f"Warning: Unexpected error parsing OpenAI response: {type(error).__name__}: {error}")
            print(f"  Response text (first 500 chars): {response_text[:500]}...")
            return []

    @staticmethod
    def _to_result(result_with_id: JobAnalysisResultWithId) -> JobAnalysisResult:
        """Convert JobAnalysisResultWithId to JobAnalysisResult."""
        return result_with_id.to_job_analysis_result()

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
