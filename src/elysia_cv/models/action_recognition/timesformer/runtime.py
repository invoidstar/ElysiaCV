"""TimeSformer runtime — build_model.  forward_train/test via common_video_forward."""
from __future__ import annotations
from typing import Any
import torch.nn as nn
from elysia_cv.models.action_recognition.common_video_forward import forward_train, forward_test

__all__ = ["build_model", "forward_train", "forward_test"]


def build_model(cfg: dict[str, Any]) -> nn.Module:
    mc = cfg.get("model", cfg)
    from elysia_cv.models.action_recognition.timesformer.model import TimeSformer
    return TimeSformer(
        model_name=str(mc.get("model_name", "facebook/timesformer-base-finetuned-k400")),
        num_classes=int(mc.get("num_classes", 101)),
    )