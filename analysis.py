#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script for analyzing job descriptions for AI skills."""

import pandas as pd

from ai_skills import find_ai_matches
from config import INPUT_CSV, OUTPUT_CSV, PREFERRED_COLUMN_ORDER
from openai_analyzer import analyze_job_desc_wrapper


def load_data() -> pd.DataFrame:
    """Load and prepare the input CSV file."""
    df = pd.read_csv(INPUT_CSV, sep=";", dtype=str, low_memory=False)
    
    # Ensure required columns exist
    df["skills"] = df["skills"].fillna("").astype(str)
    df["job_desc_text"] = df["job_desc_text"].fillna("").astype(str)
    
    return df


def analyze_skills_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the 'skills' column for AI skills using pattern matching.
    
    Args:
        df: DataFrame with 'skills' column
        
    Returns:
        DataFrame with added 'AI_skills_found' and 'AI_skill_hard' columns
    """
    # Column 1: explicit skills found (for debugging)
    df["AI_skills_found"] = df["skills"].apply(find_ai_matches)
    
    # Column 2: binary indicator
    df["AI_skill_hard"] = df["AI_skills_found"].apply(lambda s: int(bool(s)))
    
    return df


def analyze_job_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze job descriptions using OpenAI API.
    
    Args:
        df: DataFrame with 'job_desc_text' column
        
    Returns:
        DataFrame with added 'AI_skill_openai' and 'AI_skills_openai_mentioned' columns
    """
    print("Analyzing job descriptions with OpenAI...")
    print(f"Processing {len(df)} rows. This may take a while...")
    
    # Apply OpenAI analysis to job descriptions
    openai_results = df["job_desc_text"].apply(analyze_job_desc_wrapper)
    
    # Extract results into columns
    df["AI_skill_openai"] = openai_results.apply(lambda x: int(x["has_ai_skill"]))
    df["AI_skills_openai_mentioned"] = openai_results.apply(
        lambda x: ", ".join(x["ai_skills_mentioned"]) if x["ai_skills_mentioned"] else ""
    )
    
    print("OpenAI analysis complete.")
    
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to put important ones first.
    
    Args:
        df: DataFrame to reorder
        
    Returns:
        DataFrame with reordered columns
    """
    remaining_columns = [col for col in df.columns if col not in PREFERRED_COLUMN_ORDER]
    return df[PREFERRED_COLUMN_ORDER + remaining_columns]


def save_results(df: pd.DataFrame) -> None:
    """Save the results to the output CSV file."""
    df.to_csv(OUTPUT_CSV, sep=";", index=False)
    print(f"Results saved to {OUTPUT_CSV}")


def main():
    """Main execution function."""
    # Load data
    df = load_data()
    
    # Analyze skills column
    df = analyze_skills_column(df)
    
    # Analyze job descriptions with OpenAI
    df = analyze_job_descriptions(df)
    
    # Reorder columns
    df = reorder_columns(df)
    
    # Save results
    save_results(df)
    
    print("Done. Added AI_skills_found, AI_skill_hard, AI_skill_openai, and AI_skills_openai_mentioned columns.")


if __name__ == "__main__":
    main()
