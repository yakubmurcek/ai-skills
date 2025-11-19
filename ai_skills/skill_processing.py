#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level helpers for annotating AI skills in job listings."""

import pandas as pd

from .ai_skills import find_ai_matches


def annotate_declared_skills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AI-specific columns derived from the explicit `skills` field.

    Returns a copy of the input DataFrame with two new columns:
        - AI_skills_found: comma-separated list of detected AI skills
        - AI_skill_hard: binary indicator if any AI skill is present
    """
    annotated_df = df.copy()
    annotated_df["AI_skills_found"] = annotated_df["skills"].apply(find_ai_matches)
    annotated_df["AI_skill_hard"] = annotated_df["AI_skills_found"].apply(
        lambda s: int(bool(s))
    )
    return annotated_df
