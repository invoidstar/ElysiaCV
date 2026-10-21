"""TSM runtime — build_model.  forward_train/test need shift kwarg (not shared)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from elysia_cv.models.action_recognition.common_video_forward import (
    forward_train as _base_forward_train,
    forward_test as _base_forward_test,
)

__all__ = ["build_model", "forward_train", "forward_test"]


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a TSM model from config dict."""
    mc = cfg.get("model", cfg)
    from elysia_cv.models.action_recognition.tsm.model import TSM

    return TSM(
        backbone_name=str(mc.get("backbone", "resnet50")),
        num_classes=int(mc.get("num_classes", 101)),
        pretrained=bool(mc.get("pretrained", True)),
        dropout=float(mc.get("dropout", 0.5)),
        shift_div=int(mc.get("shift_div", 8)),
    )


def forward_train(model, batch, *, device=None):
    """Training forward with temporal shift."""
    return _base_forward_train(model, batch, device=device, model_kwargs={"shift": True})


def forward_test(model, batch, *, device=None):
    """Test forward without temporal shift."""
    return _base_forward_test(model, batch, device=device, model_kwargs={"shift": False})