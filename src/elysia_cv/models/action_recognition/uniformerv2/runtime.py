"""UniFormerV2 runtime — build_model.  forward_train/test via common_video_forward."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from elysia_cv.models.action_recognition.common_video_forward import (
    forward_train,
    forward_test,
)

__all__ = ["build_model", "forward_train", "forward_test"]


def build_model(cfg: dict[str, Any]) -> nn.Module:
    mc = cfg.get("model", cfg)
    from elysia_cv.models.action_recognition.uniformerv2.model import UniformerV2
    return UniformerV2(
        num_classes=int(mc.get("num_classes", 101)),
        image_size=int(mc.get("image_size", 224)),
        dropout=float(mc.get("dropout", 0.5)),
        cnn_channels=int(mc.get("cnn_channels", 64)),
        vit_backbone=str(mc.get("vit_backbone", "vit_small_patch16_224")),
        pretrained=bool(mc.get("pretrained", True)),
    )