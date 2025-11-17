#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

import pandas as pd

# ===== CONFIG =====
INPUT_CSV = "us_relevant_50.csv"
OUTPUT_CSV = "us_relevant_ai.csv"

# List of AI-related skills (case-insensitive matching)
AI_SKILLS = [
    "ai", "artificial intelligence",
    "machine learning", "ml",
    "deep learning", "dl",
    "neural network", "neural networks",
    "nlp", "natural language processing",
    "computer vision",
    "tensorflow", "pytorch", "keras",
    "scikit-learn", "sklearn",
    "hugging face", "huggingface",
    "large language model", "llm",
    "generative ai", "gen ai",
    "reinforcement learning",
    "bert", "gpt", "transformer", "transformers",
    "xgboost", "lightgbm", "catboost",
    "data science", "data scientist",
    "predictive modeling", "predictive models",
    "ai/ml", "a.i.", "a.i/ml",
    "computer vision",
    "model training", "model inference",
    "aiops"
]

# Lowercase version for reliable matching
AI_SKILLS = [skill.lower() for skill in AI_SKILLS]
AI_SKILLS_SET = set(AI_SKILLS)

# ===== LOAD CSV =====
df = pd.read_csv(INPUT_CSV, sep=";", dtype=str, low_memory=False)

# Ensure "skills" column exists
df["skills"] = df["skills"].fillna("").astype(str)

# ===== CHECK FOR AI SKILLS =====
def tokenize_skills(skill_string: str) -> list[str]:
    """Split comma-delimited skills into normalized tokens."""
    parts = re.split(r",|\n", skill_string)
    return [part.strip().lower() for part in parts if part.strip()]


def contains_ai_skills(skill_string: str) -> int:
    tokens = tokenize_skills(skill_string)
    return int(any(token in AI_SKILLS_SET for token in tokens))

df["AI_skill_hard"] = df["skills"].apply(contains_ai_skills)

# ===== REORDER COLUMNS =====
preferred_order = ["skills", "job_desc_text", "AI_skill_hard"]
remaining_columns = [col for col in df.columns if col not in preferred_order]
df = df[preferred_order + remaining_columns]

# ===== SAVE =====
df.to_csv(OUTPUT_CSV, sep=";", index=False)

print("Done. Added AI_skill_hard column.")
