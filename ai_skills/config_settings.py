#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-editable settings for the AI skills project.

Modify `get_user_config` so instructors can adjust values without touching the
main codebase. Return `None` for any value you want to fall back to the defaults
or environment variables. Because this is a regular Python module you can build
paths or names programmatically (see the output example below).

Credentials such as `OPENAI_API_KEY` must be provided through a `.env` file or
direct environment variables—they cannot be set here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def get_user_config(project_root: Path) -> Dict[str, Any]:
    """
    Return a dictionary of configuration overrides.

    Args:
        project_root: Absolute path to the project directory, passed in from
            ai_skills.config so you can compose paths relative to the repo.
    """
    data_dir = project_root / "data"
    inputs_dir = data_dir / "inputs"
    outputs_dir = data_dir / "outputs"
    input_csv = inputs_dir / "us_relevant_100.csv"
    output_filename = f"{input_csv.stem}_ai{input_csv.suffix}"
    output_csv = outputs_dir / output_filename
    # Example: generate a custom filename dynamically
    # teacher_name = "ada"
    # output_csv = outputs_dir / f"{teacher_name}_ai_results.csv"

    return {
        # --- OpenAI settings ---
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_TEMPERATURE": 0.1,
        "RATE_LIMIT_DELAY": 0.1,
        "OPENAI_BATCH_SIZE": 20,
        "OPENAI_MAX_PARALLEL_REQUESTS": 3,
        # --- Paths (edit freely or build programmatically as shown above) ---
        "DATA_DIR": data_dir,
        "INPUTS_DIR": inputs_dir,
        "OUTPUTS_DIR": outputs_dir,
        "INPUT_CSV": input_csv,
        "OUTPUT_CSV": output_csv,
        # --- Processing limits ---
        "MAX_JOB_DESC_LENGTH": 8000,
    }
