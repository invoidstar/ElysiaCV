"""TSN runtime — build_model.  forward_train/test are shared."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from elysia_cv.models.action_recognition.common_video_forward import (
    forward_train,
    forward_test,
)

__all__ = ["build_model", "forward_train", "forward_test"]


def _tsn_prep(batch: dict[str, Any]):
    """Minimal prep for TSN — just move to device."""
    images = batch["inputs"]
    labels = batch["gt_label"]
    return images, labels


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a TSN model from a config dict.

    Expected format::

        {"model": {"type": "tsn", "backbone": "resnet50",
                   "num_classes": 101, "pretrained": True, "dropout": 0.5}}
    """
    mc = cfg.get("model", cfg)

    from elysia_cv.models.action_recognition.tsn.model import TSN

    return TSN(
        backbone_name=str(mc.get("backbone", "resnet50")),
        num_classes=int(mc.get("num_classes", 101)),
        pretrained=bool(mc.get("pretrained", True)),
        dropout=float(mc.get("dropout", 0.5)),
    )