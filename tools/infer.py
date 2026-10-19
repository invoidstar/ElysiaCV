#!/usr/bin/env python3
"""Inference entry point — thin shell over ``elysia infer``.

Usage::

    python tools/infer.py configs/classification/cassava_resnet18.yaml -c best.pt -i image.jpg
    python tools/infer.py configs/detection/fcos_resnet50_coco.yaml -c best.pt -i /path/to/images/

All arguments are forwarded directly to ``elysia infer``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

if __name__ == "__main__":
    from elysia_cv.cli import main
    main(["infer"] + sys.argv[1:])