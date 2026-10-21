"""MViTv2 / Hiera runtime — build_model.  forward_train/test are shared."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from elysia_cv.models.action_recognition.common_video_forward import (
    forward_train,
    forward_test,
)


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build MViTv2 or Hiera model."""
    mc = cfg.get("model", cfg)
    from elysia_cv.models.action_recognition.mvit.model import MViT

    return MViT(
        backbone_name=str(mc.get("backbone", "mvitv2_small")),
        num_classes=int(mc.get("num_classes", 101)),
        pretrained=bool(mc.get("pretrained", True)),
        dropout=float(mc.get("dropout", 0.5)),
    )