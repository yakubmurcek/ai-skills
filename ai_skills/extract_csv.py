#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for sampling inputs before running the analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
DEFAULT_SOURCE = INPUTS_DIR / "us_relevant.csv"


def extract_sample_rows(
    *,
    rows: int,
    source_csv: Path | str | None = None,
    destination_csv: Path | str | None = None,
    sep: str = ";",
) -> Path:
    """Create a smaller CSV that keeps only the requested number of rows."""
    source = Path(source_csv) if source_csv else DEFAULT_SOURCE
    destination = (
        Path(destination_csv)
        if destination_csv
        else _build_default_destination(source, rows)
    )
    if not source.exists():
        raise FileNotFoundError(f"Source CSV not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, sep=sep, nrows=rows, encoding="utf-8")
    df.to_csv(destination, sep=sep, index=False, encoding="utf-8-sig")
    return destination


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a sample CSV for the AI Skills analyzer."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of rows to copy from the source file (default: 100).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source CSV to sample from (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help=(
            "Destination CSV (default: same directory as --source with "
            "the suffix `_<rows>`)."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI shim that mirrors the historical standalone script."""
    args = _parse_args(argv)
    destination = extract_sample_rows(
        rows=max(1, args.rows),
        source_csv=args.source,
        destination_csv=args.destination,
    )
    print(f"Done. Exported first {args.rows} rows to {destination}.")


if __name__ == "__main__":
    main()


def _build_default_destination(source: Path, rows: int) -> Path:
    return source.with_name(f"{source.stem}_{rows}{source.suffix}")
