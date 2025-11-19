#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Pydantic models used across the analysis pipeline."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobAnalysisResult(BaseModel):
    """Normalized representation of OpenAI analysis output."""

    has_ai_skill: bool = False
    ai_skills_mentioned: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, float(v)))

    def as_columns(self) -> dict:
        """Map the result to the output DataFrame columns."""
        return {
            "AI_skill_openai": int(self.has_ai_skill),
            "AI_skills_openai_mentioned": ", ".join(self.ai_skills_mentioned),
            "AI_skill_openai_confidence": self.confidence,
        }

    model_config = ConfigDict(frozen=True)  # Maintain immutability like the original dataclass


class JobAnalysisResultWithId(BaseModel):
    """Job analysis result with ID field for batch responses."""

    id: str
    has_ai_skill: bool = False
    ai_skills_mentioned: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, float(v)))

    def to_job_analysis_result(self) -> JobAnalysisResult:
        """Convert to JobAnalysisResult by removing the id field."""
        return JobAnalysisResult(
            has_ai_skill=self.has_ai_skill,
            ai_skills_mentioned=self.ai_skills_mentioned,
            confidence=self.confidence,
        )

    model_config = ConfigDict(frozen=True)


class BatchAnalysisResponse(BaseModel):
    """Response model for batch job analysis from OpenAI."""

    results: List[JobAnalysisResultWithId]

    model_config = ConfigDict(frozen=True)
