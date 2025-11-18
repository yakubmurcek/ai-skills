#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Houses editable prompt templates for OpenAI calls."""

from textwrap import dedent


def job_analysis_instructions() -> str:
    """Return the static instructions for the OpenAI job analysis prompt."""
    template = """
    Analyze job descriptions and determine if they mention AI (Artificial Intelligence),
    Machine Learning, or related skills.

    Consider these categories of AI skills:
    - Core AI/ML: artificial intelligence, machine learning, deep learning, neural networks
    - Generative AI/LLMs: GPT, BERT, LLM, large language models, generative AI, prompt engineering
    - NLP: natural language processing, text analysis, sentiment analysis
    - ML Frameworks: PyTorch, TensorFlow, Keras, scikit-learn
    - MLOps: model deployment, MLflow, model serving
    - Cloud AI: AWS SageMaker, Azure ML, Vertex AI
    - Other AI-related technologies and concepts

    Respond with a JSON object in this exact format:
    {{
        "has_ai_skill": true or false,
        "ai_skills_mentioned": ["skill1", "skill2", ...],
        "confidence": number between 0 and 1 describing how confident you are in
                       your has_ai_skill and ai_skills_mentioned answers
    }}

    Only include skills that are explicitly mentioned or clearly implied in the
    job description. Be conservative - if the job description doesn't clearly
    mention AI/ML work, set has_ai_skill to false.

    When multiple job descriptions are provided with IDs, respond with a JSON
    object that has a top-level "results" array where each element includes the
    job's ID plus the same fields above.
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
