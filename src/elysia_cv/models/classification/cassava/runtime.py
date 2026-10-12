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
    """Simple backbone + head for fine-grained classification.

    Supports two forward modes:

    - **Inference**: ``model(x)`` → ``{"logits": Tensor}``
    - **Training**: ``model(x, gt_label=labels)`` → ``{"loss": Tensor, "logits": Tensor,
      "losses": {"focal": Tensor}, "metrics": {"acc": Tensor}}``

    The training mode is designed to work directly with
    :class:`TrainerLoop._forward`, which unpacks non-input batch keys as kwargs.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self._loss_cfg: dict[str, Any] = {"type": "FocalLoss", "alpha": 0.25, "gamma": 2.0}

    def forward(self, x: torch.Tensor, gt_label: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        head_out = self.head(feats)

        # Normalize: head may return tensor or {"logits": tensor}
        if isinstance(head_out, dict):
            logits = head_out["logits"]
        else:
            logits = head_out

        if gt_label is not None:
            # ── Training path (TrainerLoop._forward calls model(x, gt_label=labels))
            loss_fn = LOSSES.build(self._loss_cfg)
            loss = loss_fn(logits, gt_label)
            pred = logits.argmax(dim=1)
            acc = (pred == gt_label).float().mean()
            return {
                "loss": loss,
                "logits": logits,
                "losses": {"focal": loss.detach()},
                "metrics": {"acc": acc.detach()},
            }

        # ── Inference path (EvalHook / forward_test)
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