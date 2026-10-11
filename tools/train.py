#!/usr/bin/env python3
"""Training entry point: ``python tools/train.py configs/xxx.yaml``."""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is on sys.path so "elysia_cv" can be imported.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

if __name__ == "__main__":
    from elysia_cv.cli import main
    # prepend "train" sub-command
    main(["train"] + sys.argv[1:])