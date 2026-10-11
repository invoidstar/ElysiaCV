#!/usr/bin/env python3
"""Evaluation entry point: ``python tools/test.py configs/xxx.yaml``."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

if __name__ == "__main__":
    from elysia_cv.cli import main
    main(["test"] + sys.argv[1:])