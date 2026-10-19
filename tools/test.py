#!/usr/bin/env python3
"""Evaluation entry point — thin shell over ``elysia test``.

Usage::

    python tools/test.py configs/classification/cassava_resnet18.yaml
    python tools/test.py configs/detection/fcos_resnet50_coco.yaml --checkpoint path/to/best.pt

All arguments are forwarded directly to ``elysia test``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

if __name__ == "__main__":
    from elysia_cv.cli import main
    main(["test"] + sys.argv[1:])