"""ST-Adapter runtime — build_model.  forward_train/test via common_video_forward."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from elysia_cv.models.action_recognition.common_video_forward import (
    forward_train,
    forward_test,
)

__all__ = ["build_model", "forward_train", "forward_test"]


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build ST-Adapter from config dict."""
    mc = cfg.get("model", cfg)
    from elysia_cv.models.action_recognition.st_adapter.model import STAdapter

    return STAdapter(
        backbone_name=str(mc.get("backbone", "vit_base_patch16_clip_224")),
        num_classes=int(mc.get("num_classes", 101)),
        pretrained=bool(mc.get("pretrained", True)),
        adapter_ratio=int(mc.get("adapter_ratio", 4)),
        dropout=float(mc.get("dropout", 0.5)),
        freeze_backbone=bool(mc.get("freeze_backbone", True)),
    )