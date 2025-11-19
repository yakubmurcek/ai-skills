#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy wrapper that now delegates to the unified CLI."""

from __future__ import annotations

import sys
from typing import Iterable

from ai_skills import cli


def main(argv: Iterable[str] | None = None) -> int:
    """Run the CLI, defaulting to the analyze command for compatibility."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["analyze"]
    return cli.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
