#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Houses editable prompt templates for OpenAI calls."""

from textwrap import dedent


def job_analysis_instructions() -> str:
    """Return the static instructions for the OpenAI job analysis prompt."""
    template = """
    You are classifying whether a job actually involves AI/ML work.

Definition:
A job “involves AI/ML work” only if the role requires building, training, fine-tuning,
evaluating, deploying, integrating, or maintaining machine learning or deep learning
models, LLMs, or other AI systems.

Do NOT classify as AI/ML work:
- roles that only mention AI as a company product ("AI-powered platform")
- roles that only use AI tools indirectly (e.g., analysts using dashboards)
- generic software engineering roles where AI terms appear in marketing text
- roles that only mention buzzwords without concrete ML tasks

Examples of AI/ML work:
- “train and deploy ML models”
- “build predictive models using scikit-learn”
- “fine-tune LLMs for production”
- “design data pipelines for model training”
- “implement inference services with PyTorch/TensorFlow/Vertex AI”
- “develop RAG pipelines”

Examples of NOT AI/ML work:
- “React developer for an AI startup”
- “maintain backend services for AI product”
- “AI is part of company vision”
- “work with data (no modeling tasks)”
- “general software engineering with no modeling responsibilities”

HARD RULES (CRITICAL):

1. The presence of the words “AI”, “AI-powered”, “AI startup”, or “machine learning company” 
   does NOT imply the job involves AI/ML work.

2. General software engineering tasks are NEVER AI skills. This includes:
   - building or shipping web features
   - designing APIs
   - writing backend or frontend code (React, Flask, Node, SQL, APIs, etc.)
   - improving architecture
   - maintaining infrastructure
   - DevOps, testing, monitoring, product thinking, agile collaboration
   - anything that does not explicitly describe ML model work

3. If the description does NOT mention tasks like:
   - training machine learning models
   - fine-tuning LLMs
   - designing model architectures
   - building inference pipelines
   - deploying ML models
   - evaluating ML performance
   - working directly with ML frameworks (PyTorch, TensorFlow, etc.)
   Then: 
       has_ai_skill = false
       ai_skills_mentioned = []

4. Never infer AI skills from:
   - company description
   - product marketing
   - the fact that the company uses AI internally
   - general engineering responsibilities

5. If no explicit model-related tasks are present, return false even if:
   - the company is “AI-powered”
   - the product uses AI
   - the role is a software engineer at an AI company

Output JSON example:
{
  "ai_job": true or false,
  "rationale": "short explanation",
  "confidence": 0-1
}
    """
    return dedent(template).strip()


def job_analysis_prompt(job_description: str) -> str:
    """Generate the user prompt that only contains the job description text."""
    template = """
    Job Description:
    {job_description}
    """
    return dedent(template).strip().format(job_description=job_description)


def job_analysis_batch_prompt(batch_items: list[tuple[str, str]]) -> str:
    """Build a prompt covering multiple job descriptions in a single request."""
    header = dedent(
        """
        Analyze each of the following job descriptions independently.
        Return a JSON object with a top-level `results` array.
        Each entry must include: id, has_ai_skill, ai_skills_mentioned, confidence.
        """
    ).strip()

    body_parts = []
    for identifier, description in batch_items:
        body_parts.append(
            dedent(
                f"""
                Job id: {identifier}
                Job Description:
                {description}
                """
            ).strip()
        )
    return f"{header}\n\n" + "\n\n".join(body_parts)
