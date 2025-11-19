#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry point for the AI skills analysis pipeline."""

import threading
import sys
import time

from src.config import OUTPUT_CSV
from src.pipeline import JobAnalysisPipeline


def main():
    """Execute the refactored analysis pipeline with a CLI progress bar."""
    pipeline = JobAnalysisPipeline()
    progress = _CLIProgressBar("Analyzing job descriptions")

    start_time = time.perf_counter()
    try:
        df = pipeline.run(progress_callback=progress.update)
    finally:
        progress.finish()
    elapsed = time.perf_counter() - start_time

    print(
        f"Processed {len(df)} job descriptions. "
        f"Added AI columns and saved results to {OUTPUT_CSV}. "
        f"(Elapsed: {elapsed:.1f}s)"
    )


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
    main()
