"""Cassava Leaf Disease runtime — build / train / test bridge.

Fine-grained classification with FocalLoss for class imbalance.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import BACKBONES, HEADS, LOSSES


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build cassava classification model from config.

    Layout: Backbone (ResNet) → Head (LinearClsHead)
    """
    model_cfg = cfg.get("model", cfg)

    backbone_cfg = dict(model_cfg.get("backbone", {}))
    backbone_type = backbone_cfg.pop("type", "ResNet")
    backbone = BACKBONES.build({"type": backbone_type, **backbone_cfg})

    head_cfg = dict(model_cfg.get("head", {}))
    head_type = head_cfg.pop("type", "LinearClsHead")
    head = HEADS.build({"type": head_type, **head_cfg})

    model = CassavaModel(backbone=backbone, head=head)

    pretrained = model_cfg.get("pretrained_path")
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state, strict=False)

    return model


class CassavaModel(nn.Module):
    """Simple backbone + head for fine-grained classification."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        logits = self.head(feats)
        return {"logits": logits}


def forward_train(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """One training forward pass with FocalLoss for class imbalance.

    Expects batch keys: ``inputs`` (images), ``gt_label`` (class labels).
    """
    images = batch["inputs"]
    labels = batch.get("gt_label")
    if device is not None:
        images = images.to(device, non_blocking=True)
        if labels is not None:
            labels = labels.to(device, non_blocking=True)

    outputs = model(images)
    logits = outputs["logits"]

    # FocalLoss to handle class imbalance (CMD is 61.5% of data)
    loss_fn = LOSSES.build({"type": "FocalLoss", "alpha": 0.25, "gamma": 2.0})
    loss = loss_fn(logits, labels)

    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean()

    return {
        "loss": loss,
        "losses": {"focal": loss.detach()},
        "metrics": {"acc": acc.detach()},
    }


def forward_test(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Eval forward pass."""
    images = batch["inputs"]
    if device is not None:
        images = images.to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(images)
    return {"logits": outputs["logits"].detach()}


def get_default_config() -> dict[str, Any]:
    return {
        "model": {
            "type": "cassava_resnet50",
            "backbone": {"type": "ResNet", "depth": 50},
            "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 2048},
        }
    }