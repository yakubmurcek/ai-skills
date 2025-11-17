#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

import pandas as pd

# ===== CONFIG =====
INPUT_CSV = "us_relevant_50.csv"
OUTPUT_CSV = "us_relevant_ai.csv"

# List of AI-related skills (case-insensitive matching)
AI_SKILLS = [
    # Core AI / ML terms
    "ai", "artificial intelligence",
    "ml", "machine learning",
    "dl", "deep learning",
    "neural network", "neural networks",
    "supervised learning", "unsupervised learning",
    "reinforcement learning", "rl",
    "self-supervised learning",
    "classification", "regression", "clustering",

    # GenAI / LLMs
    "llm", "large language model", "large-language model",
    "gpt", "gpt-3", "gpt-4", "gpt4", "gpt3",
    "bert", "roberta", "distilbert", "albert",
    "t5", "llama", "mistral",
    "generative ai", "genai", "gen ai",
    "text generation", "text-generation",
    "prompt engineering",

    # Vision / Speech
    "computer vision",
    "image recognition", "object detection",
    "yolo", "retinanet", "mask r-cnn", "rcnn",
    "speech recognition",

    # NLP
    "nlp", "natural language processing",
    "nlu", "natural language understanding",
    "nlg", "natural language generation",
    "tokenization", "word embeddings",
    "word2vec", "glove", "fasttext",
    "named entity recognition", "ner",
    "sentiment analysis",

    # ML Frameworks
    "pytorch", "torch",
    "tensorflow", "tf", "keras",
    "jax", "flax",
    "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost",

    # Data science & model building
    "model training", "model inference",

    # MLOps / deployment / pipelines
    "mlops", "aiops",
    "model deployment", "model serving",
    "model monitoring", "model governance",
    "feature store",
    "mlflow", "kubeflow",
    "torchserve", "tensorflow serving",
    "onnx", "onnxruntime",

    # Cloud AI tooling
    "sageMaker", "sagemaker",
    "vertex ai",
    "azure ml", "azure machine learning",
    "aws ai", "gcp ai",

    # GPU / training infra
    "cuda", "cudnn",
    "gpu acceleration", "gpu-accelerated",
    "distributed training",

    # Data processing (ML-relevant)
    "pandas", "numpy", "scipy",
    "spark ml", "pyspark", "sparkml",
    "data pipeline", "etl pipeline",

    # Edge AI
    "tflite", "tensorflow lite",
    "coreml",
    "edge ai", "edge ml"
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

def find_ai_matches(skill_string: str) -> str:
    """Return comma-separated AI skills found in the string."""
    tokens = tokenize_skills(skill_string)
    matches = sorted(set(t for t in tokens if t in AI_SKILLS_SET))
    return ", ".join(matches)

# Column 1: explicit skills found (for debugging)
df["AI_skills_found"] = df["skills"].apply(find_ai_matches)

# Column 2: binary indicator
df["AI_skill_hard"] = df["AI_skills_found"].apply(lambda s: int(bool(s)))

# ===== REORDER COLUMNS =====
preferred_order = ["skills", "job_desc_text", "AI_skills_found", "AI_skill_hard"]
remaining_columns = [col for col in df.columns if col not in preferred_order]
df = df[preferred_order + remaining_columns]

# ===== SAVE =====
df.to_csv(OUTPUT_CSV, sep=";", index=False)

print("Done. Added AI_skills_found and AI_skill_hard columns.")
