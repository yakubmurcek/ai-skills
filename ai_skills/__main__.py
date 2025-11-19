#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Allow `python -m src` to invoke the CLI directly."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
