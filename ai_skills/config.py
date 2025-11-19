#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration settings for the AI skills analysis project.

Edit `config_settings.py` to customize paths, OpenAI parameters, or limits.
Values from that file take precedence over environment variables or `.env`,
except for credentials such as `OPENAI_API_KEY`, which must come from the
environment for security reasons.
This module still exposes three structured sections:

* `PATHS` - where data is read/written
* `OPENAI` - how the OpenAI client is configured
* `PROCESSING` - limits and column ordering for downstream steps

The legacy module-level constants (e.g., `INPUT_CSV`, `OPENAI_MODEL`) remain
available for compatibility with existing imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

# Resolve project root and load environment variables
PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

try:
    from .config_settings import get_user_config
except ImportError:
    def get_user_config(_: Path) -> Dict[str, Any]:
        """Fallback when config_settings.py is absent."""
        return {}


USER_CONFIG = get_user_config(PROJECT_ROOT)


def _get_raw_setting(env_var: str) -> Optional[Any]:
    """Return the first non-None value among user overrides and env vars."""
    if env_var in USER_CONFIG:
        value = USER_CONFIG[env_var]
        if value is not None:
            return value
    value = os.getenv(env_var)
    if value is not None and value.strip() != "":
        return value.strip()
    return None


def _get_env_only_setting(env_var: str) -> Optional[str]:
    """Read sensitive values exclusively from the environment or .env."""
    value = os.getenv(env_var)
    if value is not None and value.strip() != "":
        return value.strip()
    return None


def _get_int_setting(env_var: str, default: int) -> int:
    """Safely parse integer settings from the environment."""
    value = _get_raw_setting(env_var)
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_setting(env_var: str, default: float) -> float:
    """Safely parse float settings from the environment."""
    value = _get_raw_setting(env_var)
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return default


def _get_path_setting(env_var: str, default: Path) -> Path:
    """Return a path overridden by env var if provided."""
    value = _get_raw_setting(env_var)
    if value:
        if isinstance(value, Path):
            return value.expanduser().resolve()
        return Path(str(value)).expanduser().resolve()
    return default


@dataclass(frozen=True)
class PathSettings:
    """All file-system locations used by the pipeline."""

    project_root: Path
    data_dir: Path
    inputs_dir: Path
    outputs_dir: Path
    input_csv: Path
    output_csv: Path

    @classmethod
    def build(cls, project_root: Path) -> "PathSettings":
        data_dir = _get_path_setting("DATA_DIR", project_root / "data")
        inputs_dir = _get_path_setting("INPUTS_DIR", data_dir / "inputs")
        outputs_dir = _get_path_setting("OUTPUTS_DIR", data_dir / "outputs")
        input_csv = _get_path_setting(
            "INPUT_CSV", inputs_dir / "us_relevant_50.csv"
        )
        output_csv = _get_path_setting(
            "OUTPUT_CSV", outputs_dir / "us_relevant_ai_50.csv"
        )
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            inputs_dir=inputs_dir,
            outputs_dir=outputs_dir,
            input_csv=input_csv,
            output_csv=output_csv,
        )


@dataclass(frozen=True)
class OpenAISettings:
    """Options that control OpenAI API usage."""

    api_key: str
    model: str
    temperature: float
    rate_limit_delay: float
    batch_size: int
    max_parallel_requests: int

    @classmethod
    def build(cls) -> "OpenAISettings":
        api_key = _get_env_only_setting("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file or as an environment variable."
            )

        return cls(
            api_key=api_key,
            model=_get_raw_setting("OPENAI_MODEL") or "gpt-4o-mini",
            temperature=_get_float_setting("OPENAI_TEMPERATURE", 0.1),
            rate_limit_delay=_get_float_setting("RATE_LIMIT_DELAY", 0.1),
            batch_size=max(1, _get_int_setting("OPENAI_BATCH_SIZE", 20)),
            max_parallel_requests=max(
                1, _get_int_setting("OPENAI_MAX_PARALLEL_REQUESTS", 3)
            ),
        )


DEFAULT_COLUMN_ORDER: Tuple[str, ...] = (
    "skills",
    "job_desc_text",
    "AI_skills_found",
    "AI_skill_hard",
    "AI_skill_openai",
    "AI_skill_openai_confidence",
    "AI_skills_openai_mentioned",
)


@dataclass(frozen=True)
class ProcessingSettings:
    """Tunable processing limits and output ordering."""

    max_job_desc_length: int
    preferred_column_order: Tuple[str, ...]

    @classmethod
    def build(cls) -> "ProcessingSettings":
        return cls(
            max_job_desc_length=max(1, _get_int_setting("MAX_JOB_DESC_LENGTH", 8000)),
            preferred_column_order=DEFAULT_COLUMN_ORDER,
        )


@dataclass(frozen=True)
class AppConfig:
    """Aggregate configuration for consumers that prefer structured access."""

    paths: PathSettings
    openai: OpenAISettings
    processing: ProcessingSettings


PATHS = PathSettings.build(PROJECT_ROOT)
OPENAI = OpenAISettings.build()
PROCESSING = ProcessingSettings.build()
CONFIG = AppConfig(paths=PATHS, openai=OPENAI, processing=PROCESSING)

# Backwards-compatible module-level exports
DATA_DIR = PATHS.data_dir
INPUTS_DIR = PATHS.inputs_dir
OUTPUTS_DIR = PATHS.outputs_dir
INPUT_CSV = PATHS.input_csv
OUTPUT_CSV = PATHS.output_csv
OPENAI_API_KEY = OPENAI.api_key
OPENAI_MODEL = OPENAI.model
OPENAI_TEMPERATURE = OPENAI.temperature
RATE_LIMIT_DELAY = OPENAI.rate_limit_delay
OPENAI_BATCH_SIZE = OPENAI.batch_size
OPENAI_MAX_PARALLEL_REQUESTS = OPENAI.max_parallel_requests
MAX_JOB_DESC_LENGTH = PROCESSING.max_job_desc_length
PREFERRED_COLUMN_ORDER = list(PROCESSING.preferred_column_order)

# List of AI-related skills (case-insensitive matching)
AI_SKILLS = [
    # Core AI / ML terms
    "ai",
    "artificial intelligence",
    "ml",
    "machine learning",
    "dl",
    "deep learning",
    "neural network",
    "neural networks",
    "supervised learning",
    "unsupervised learning",
    "reinforcement learning",
    "rl",
    "self-supervised learning",
    "classification",
    "regression",
    "clustering",
    # GenAI / LLMs
    "llm",
    "large language model",
    "large-language model",
    "gpt",
    "gpt-3",
    "gpt-4",
    "gpt4",
    "gpt3",
    "bert",
    "roberta",
    "distilbert",
    "albert",
    "t5",
    "llama",
    "mistral",
    "generative ai",
    "genai",
    "gen ai",
    "text generation",
    "text-generation",
    "prompt engineering",
    "prompt"
    # Vision / Speech
    "computer vision",
    "image recognition",
    "object detection",
    "yolo",
    "retinanet",
    "mask r-cnn",
    "rcnn",
    "speech recognition",
    # NLP
    "nlp",
    "natural language processing",
    "nlu",
    "natural language understanding",
    "nlg",
    "natural language generation",
    "tokenization",
    "word embeddings",
    "word2vec",
    "glove",
    "fasttext",
    "named entity recognition",
    "ner",
    "sentiment analysis",
    # ML Frameworks
    "pytorch",
    "torch",
    "tensorflow",
    "tf",
    "keras",
    "jax",
    "flax",
    "scikit-learn",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    # Data science & model building
    "model training",
    "model inference",
    # MLOps / deployment / pipelines
    "mlops",
    "aiops",
    "model deployment",
    "model serving",
    "model monitoring",
    "model governance",
    "feature store",
    "mlflow",
    "kubeflow",
    "torchserve",
    "tensorflow serving",
    "onnx",
    "onnxruntime",
    # Cloud AI tooling
    "sageMaker",
    "sagemaker",
    "vertex ai",
    "azure ml",
    "azure machine learning",
    "aws ai",
    "gcp ai",
    # GPU / training infra
    "cuda",
    "cudnn",
    "gpu acceleration",
    "gpu-accelerated",
    "distributed training",
    # Data processing (ML-relevant)
    "pandas",
    "numpy",
    "scipy",
    "spark ml",
    "pyspark",
    "sparkml",
    "data pipeline",
    "etl pipeline",
    # Edge AI
    "tflite",
    "tensorflow lite",
    "coreml",
    "edge ai",
    "edge ml",
]
