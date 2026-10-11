"""Metrics — auto-registered via METRICS registry."""

from __future__ import annotations

from elysia_cv.metrics.accuracy import Accuracy, Top5Accuracy  # noqa: F401
from elysia_cv.metrics.iou import MeanIoU, MeanAccuracy, MeanAP  # noqa: F401