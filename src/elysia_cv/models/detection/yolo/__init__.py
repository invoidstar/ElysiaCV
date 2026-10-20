"""YOLO detection models — pure PyTorch implementation, zero ultralytics dependency.

Supports YOLOv10/v11/v12/YOLO26 across n/s/m/l/x variants (18 total).

Quick start::

    from elysia_cv.models.detection.yolo import YOLODetector, build_yolo

    model = build_yolo("yolov11n", num_classes=80)
    out = model(torch.randn(1, 3, 640, 640))
"""

from .model import YOLODetector, build_yolo
from .variants import get_variant, list_variants
from .runtime import build_model, forward_train, forward_test, infer
from .losses import YOLOv8Loss, V10Loss
from .hdms_yolo import HDMSDetector, build_hdms
from .weights import download_weights, load_weights, list_available_weights

__all__ = [
    "YOLODetector",
    "HDMSDetector",
    "build_yolo",
    "build_hdms",
    "build_model",
    "forward_train",
    "forward_test",
    "infer",
    "get_variant",
    "list_variants",
    "YOLOv8Loss",
    "V10Loss",
    "download_weights",
    "load_weights",
    "list_available_weights",
]