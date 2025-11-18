#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration settings for the AI skills analysis project."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root and load environment variables
PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# File paths (always relative to the project directory)
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
INPUT_CSV = INPUTS_DIR / "us_relevant_10.csv"
OUTPUT_CSV = OUTPUTS_DIR / "us_relevant_ai_10.csv"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please set it in your .env file or as an environment variable."
    )

# OpenAI model settings
OPENAI_MODEL = "gpt-4o-mini"  # Can be changed to "gpt-4o" if needed
OPENAI_TEMPERATURE = 0.1
RATE_LIMIT_DELAY = 0.1  # Seconds between API calls
OPENAI_BATCH_SIZE = 5  # Number of job descriptions analyzed per API request

# Text processing limits
MAX_JOB_DESC_LENGTH = 8000  # Characters to truncate long descriptions

# Column ordering
PREFERRED_COLUMN_ORDER = [
    "skills",
    "job_desc_text",
    "AI_skills_found",
    "AI_skill_hard",
    "AI_skill_openai",
    "AI_skill_openai_confidence",
    "AI_skills_openai_mentioned"
]

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
