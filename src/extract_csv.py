#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility to create a smaller sample file from the main CSV."""

from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent
INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"

extract_rows = 100
input_csv = INPUTS_DIR / "us_relevant.csv"
output_csv = INPUTS_DIR / f"us_relevant_{extract_rows}.csv"

INPUTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_csv, sep=";", nrows=extract_rows, encoding="utf-8")

df.to_csv(output_csv, sep=";", index=False, encoding="utf-8-sig")

print(f"Done. Exported first {extract_rows} rows to {output_csv}.")
