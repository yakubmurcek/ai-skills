#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level orchestration of the AI skill analysis workflow."""

from __future__ import annotations

import pandas as pd
from typing import Callable

from .data_io import load_input_data, reorder_columns, save_results
from .models import JobAnalysisResult
from .openai_analyzer import OpenAIJobAnalyzer
from .skill_processing import annotate_declared_skills


class JobAnalysisPipeline:
    """Coordinates each step of data ingestion and enrichment."""

    def __init__(self, *, analyzer: OpenAIJobAnalyzer | None = None) -> None:
        self.analyzer = analyzer or OpenAIJobAnalyzer()

    def run(
        self, *, progress_callback: Callable[[int, int], None] | None = None
    ) -> pd.DataFrame:
        """Execute the full pipeline and return the final DataFrame."""
        df = load_input_data()
        df = annotate_declared_skills(df)
        df = self._annotate_job_descriptions(df, progress_callback)
        df = reorder_columns(df)
        save_results(df)
        return df

    def _annotate_job_descriptions(
        self, df: pd.DataFrame, progress_callback: Callable[[int, int], None] | None = None
    ) -> pd.DataFrame:
        """Apply the OpenAI analyzer to every job description."""
        annotated_df = df.copy()
        job_texts = [
            None if pd.isna(text) else str(text)
            for text in annotated_df["job_desc_text"].tolist()
        ]
        results = self.analyzer.analyze_texts(
            job_texts, progress_callback=progress_callback
        )
        annotated_df["AI_skill_openai"] = [self._as_indicator(r) for r in results]
        annotated_df["AI_skills_openai_mentioned"] = [
            self._as_joined_skills(r) for r in results
        ]
        annotated_df["AI_skill_openai_confidence"] = [
            self._as_confidence(r) for r in results
        ]
        return annotated_df

    @staticmethod
    def _as_indicator(result: JobAnalysisResult) -> int:
        return int(result.has_ai_skill)

    @staticmethod
    def _as_joined_skills(result: JobAnalysisResult) -> str:
        return ", ".join(result.ai_skills_mentioned)

    @staticmethod
    def _as_confidence(result: JobAnalysisResult) -> float:
        return result.confidence
