#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI API integration for analyzing job descriptions."""

import json
import time
from typing import Dict, List

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    RATE_LIMIT_DELAY,
    MAX_JOB_DESC_LENGTH,
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def analyze_job_desc_with_openai(job_desc_text: str) -> Dict[str, any]:
    """
    Use OpenAI to analyze job description text and determine if it contains AI skills.
    
    Args:
        job_desc_text: The job description text to analyze
        
    Returns:
        Dict with 'has_ai_skill' (bool) and 'ai_skills_mentioned' (list of strings)
    """
    if not job_desc_text or not job_desc_text.strip():
        return {"has_ai_skill": False, "ai_skills_mentioned": []}
    
    # Truncate very long descriptions to avoid token limits
    text_to_analyze = (
        job_desc_text[:MAX_JOB_DESC_LENGTH]
        if len(job_desc_text) > MAX_JOB_DESC_LENGTH
        else job_desc_text
    )
    
    prompt = f"""Analyze the following job description and determine if it mentions any AI (Artificial Intelligence), Machine Learning, or related skills.

Consider these categories of AI skills:
- Core AI/ML: artificial intelligence, machine learning, deep learning, neural networks
- Generative AI/LLMs: GPT, BERT, LLM, large language models, generative AI, prompt engineering
- NLP: natural language processing, text analysis, sentiment analysis
- Computer Vision: image recognition, object detection, computer vision
- ML Frameworks: PyTorch, TensorFlow, Keras, scikit-learn
- MLOps: model deployment, MLflow, model serving
- Cloud AI: AWS SageMaker, Azure ML, Vertex AI
- Other AI-related technologies and concepts

Job Description:
{text_to_analyze}

Respond with a JSON object in this exact format:
{{
    "has_ai_skill": true or false,
    "ai_skills_mentioned": ["skill1", "skill2", ...]
}}

Only include skills that are explicitly mentioned or clearly implied in the job description. Be conservative - if the job description doesn't clearly mention AI/ML work, set has_ai_skill to false."""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing job descriptions for AI and machine learning skills. Always respond with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=OPENAI_TEMPERATURE,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        # Validate and normalize the response
        has_ai = bool(result.get("has_ai_skill", False))
        skills_list = result.get("ai_skills_mentioned", [])
        if not isinstance(skills_list, list):
            skills_list = []
        
        return {
            "has_ai_skill": has_ai,
            "ai_skills_mentioned": skills_list
        }
    
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse OpenAI JSON response: {e}")
        return {"has_ai_skill": False, "ai_skills_mentioned": []}
    except Exception as e:
        print(f"Warning: OpenAI API error: {e}")
        return {"has_ai_skill": False, "ai_skills_mentioned": []}


def analyze_job_desc_wrapper(job_desc_text: str) -> Dict[str, any]:
    """
    Wrapper function with rate limiting and error handling.
    
    Args:
        job_desc_text: The job description text to analyze
        
    Returns:
        Dict with 'has_ai_skill' (bool) and 'ai_skills_mentioned' (list of strings)
    """
    result = analyze_job_desc_with_openai(job_desc_text)
    # Small delay to avoid rate limits
    time.sleep(RATE_LIMIT_DELAY)
    return result

