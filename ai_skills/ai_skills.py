#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI skills matching functions."""

import re
from typing import List, Set

from .config import AI_SKILLS

# Lowercase version for reliable matching
AI_SKILLS_LOWER = [skill.lower() for skill in AI_SKILLS]
AI_SKILLS_SET: Set[str] = set(AI_SKILLS_LOWER)


def tokenize_skills(skill_string: str) -> List[str]:
    """Split comma-delimited skills into normalized tokens."""
    parts = re.split(r",|\n", skill_string)
    return [part.strip().lower() for part in parts if part.strip()]


def find_ai_matches(skill_string: str) -> str:
    """Return comma-separated AI skills found in the string."""
    tokens = tokenize_skills(skill_string)
    matches = sorted(set(t for t in tokens if t in AI_SKILLS_SET))
    return ", ".join(matches)
