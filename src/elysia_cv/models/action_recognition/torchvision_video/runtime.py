"""Torchvision video runtime — build_model.  forward_train/test via common_video_forward (NCDHW prep)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from elysia_cv.models.action_recognition.common_video_forward import (
    forward_train as _base_forward_train,
    forward_test as _base_forward_test,
)

__all__ = ["build_model", "forward_train", "forward_test"]


def _prep_ncdhw(batch: dict[str, Any]):
    """Preprocess batch into NCDHW [B, C, T, H, W] format for torchvision video models."""
    images = batch["inputs"]
    labels = batch["gt_label"]
    # Collapse DataLoader-duplicate batch dim
    if images.dim() == 6:
        images = images.view(-1, *images.shape[2:])
    # Convert to NCDHW
    if images.dim() == 4:
        images = images.unsqueeze(0).permute(0, 2, 1, 3, 4)
    elif images.dim() == 5 and images.shape[1] != 3 and images.shape[2] == 3:
        images = images.permute(0, 2, 1, 3, 4)
    return images, labels


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a torchvision video model from config dict."""
    mc = cfg.get("model", cfg)
    from elysia_cv.models.action_recognition.torchvision_video.model import TorchvisionVideo

    return TorchvisionVideo(
        backbone_name=str(mc.get("backbone", "r3d_18")),
        num_classes=int(mc.get("num_classes", 101)),
        pretrained=bool(mc.get("pretrained", True)),
    )


def forward_train(model, batch, *, device=None):
    return _base_forward_train(model, batch, device=device, prep_fn=_prep_ncdhw)


def forward_test(model, batch, *, device=None):
    return _base_forward_test(model, batch, device=device, prep_fn=_prep_ncdhw)