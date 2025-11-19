#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entry point for preparing data and running the AI skills pipeline."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from .extract_csv import DEFAULT_SOURCE, extract_sample_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-skills",
        description="Prepare inputs and analyze job descriptions for AI skills.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser(
        "prepare-inputs",
        help="Create a smaller CSV sample for experimentation or grading.",
    )
    prep.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of rows to copy from the source CSV (default: 100).",
    )
    prep.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source CSV to sample from (default: {DEFAULT_SOURCE}).",
    )
    prep.add_argument(
        "--destination",
        type=Path,
        default=None,
        help=(
            "Destination CSV that the pipeline will read "
            "(default: derive from --source and the selected row count)."
        ),
    )
    prep.add_argument(
        "--sep",
        type=str,
        default=";",
        help="Column separator for both files (default: ';').",
    )
    prep.set_defaults(func=_handle_prepare_inputs)

    analyze = sub.add_parser(
        "analyze",
        help="Run the full OpenAI-powered pipeline on the configured CSV.",
    )
    analyze.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the CLI progress bar (useful when piping output).",
    )
    analyze.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Override the CSV file to analyze instead of the configured INPUT_CSV.",
    )
    analyze.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write results to a custom path (default: derived from config or --input-csv).",
    )
    analyze.set_defaults(func=_handle_analyze)

    return parser


def _handle_prepare_inputs(args: argparse.Namespace) -> int:
    destination = extract_sample_rows(
        rows=max(1, args.rows),
        source_csv=args.source,
        destination_csv=args.destination,
        sep=args.sep,
    )
    print(
        f"Created sample of {args.rows} rows at {destination}. "
        "Update INPUT_CSV in config if you choose a custom destination."
    )
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    from .config import OUTPUT_CSV, PATHS  # Imported lazily to avoid requiring an API key early.
    from .pipeline import JobAnalysisPipeline

    pipeline = JobAnalysisPipeline()
    progress = None
    callback: Callable[[int, int], None] | None = None
    if not args.no_progress and sys.stderr.isatty():
        progress = _CLIProgressBar("Analyzing job descriptions")
        callback = progress.update

    output_path: Path | None = args.output_csv
    if output_path is None and args.input_csv is not None:
        input_path = Path(args.input_csv)
        output_path = PATHS.outputs_dir / f"{input_path.stem}_ai{input_path.suffix}"

    start_time = time.perf_counter()
    try:
        df = pipeline.run(
            progress_callback=callback,
            input_csv=args.input_csv,
            output_csv=output_path,
        )
    finally:
        if progress:
            progress.finish()
    elapsed = time.perf_counter() - start_time

    print(
        f"Processed {len(df)} job descriptions. "
        f"Added AI columns and saved results to {(output_path or OUTPUT_CSV)}. "
        f"(Elapsed: {elapsed:.1f}s)"
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - CLI convenience branch
        parser.exit(1, f"Error: {exc}\n")


class _CLIProgressBar:
    """ASCII progress bar with inline percentage and spinner animation."""

    def __init__(
        self, message: str, width: int = 40, refresh_interval: float = 0.1
    ) -> None:
        self.message = message
        self.width = max(10, width)
        self.current = 0
        self.total = 0
        self._active = True
        self._refresh_interval = max(0.05, refresh_interval)
        self._fill_char = "█"
        self._empty_char = "░"
        self._spinner_frames = "|/-\\"
        self._spinner_index = 0
        self._render_lock = threading.Lock()
        self._last_line_len = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def update(self, completed: int, total: int) -> None:
        if not self._active:
            return
        self.total = max(total, 0)
        safe_total = max(self.total, 1)
        self.current = max(0, min(completed, safe_total))
        self._render()

    def finish(self) -> None:
        if not self._active:
            return
        self.current = max(self.current, self.total)
        self._stop_event.set()
        self._thread.join()
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._last_line_len = 0
        self._active = False

    def _animate(self) -> None:
        while not self._stop_event.is_set():
            self._render(tick=True)
            time.sleep(self._refresh_interval)

    def _render(self, *, tick: bool = False) -> None:
        if not self._active:
            return
        with self._render_lock:
            if tick:
                self._spinner_index = (self._spinner_index + 1) % len(
                    self._spinner_frames
                )
            spinner = self._spinner_frames[self._spinner_index]
            safe_total = max(self.total, 1)
            completed = max(0, min(self.current, safe_total))
            percent = completed / safe_total
            filled = min(self.width, int(round(percent * self.width)))
            bar_chars = [self._empty_char] * self.width
            for i in range(filled):
                bar_chars[i] = self._fill_char

            percent_text = f"{percent * 100:5.1f}%"
            start = max(0, (self.width - len(percent_text)) // 2)
            for index, ch in enumerate(percent_text):
                pos = start + index
                if pos < self.width:
                    bar_chars[pos] = ch

            bar = "".join(bar_chars)
            total_label = "?" if self.total == 0 else str(self.total)
            line = (
                f"{self.message} {spinner} [{bar}] {completed}/{total_label}"
            )
            self._write_line(line)

    def _write_line(self, line: str) -> None:
        pad = max(0, self._last_line_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_line_len = len(line)


if __name__ == "__main__":
    raise SystemExit(main())
