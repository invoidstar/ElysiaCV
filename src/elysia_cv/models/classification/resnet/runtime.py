"""ResNet classification runtime — bridge between engine and model.

Provides ``build_model``, ``forward_train``, and ``forward_test``
that the training engine calls.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from elysia_cv.core.registry import BACKBONES, HEADS, LOSSES


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a ResNet classification model from a config dict.

    Expects ``cfg.model.backbone``, ``cfg.model.head``, ``cfg.model.pretrained_path``.
    """
    model_cfg = cfg.get("model", cfg)

    backbone_cfg = dict(model_cfg.get("backbone", {}))
    backbone_type = backbone_cfg.pop("type", "ResNet")
    backbone = BACKBONES.build({"type": backbone_type, **backbone_cfg})

    head_cfg = dict(model_cfg.get("head", {}))
    head_type = head_cfg.pop("type", "LinearClsHead")
    head = HEADS.build({"type": head_type, **head_cfg})

    model = ClassificationModel(backbone=backbone, head=head)

    # Load pretrained weights if specified
    pretrained = model_cfg.get("pretrained_path")
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state, strict=False)

    return model


class ClassificationModel(nn.Module):
    """Simple backbone + head classification model."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        return {"logits": logits}


def forward_train(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run one training forward pass; returns ``loss`` + per-objective losses.

    Expects batch keys: ``inputs`` (images), ``gt_label`` (class labels).
    Supports MixUp via ``gt_label_soft`` (soft one-hot targets).
    """
    import torch.nn.functional as F
    images = batch["inputs"]
    labels = batch.get("gt_label")
    if labels is not None and hasattr(labels, "to") and device is not None:
        labels = labels.to(device, non_blocking=True)

    labels_soft = batch.get("gt_label_soft")
    if labels_soft is not None and hasattr(labels_soft, "to") and device is not None:
        labels_soft = labels_soft.to(device, non_blocking=True)

    if device is not None:
        images = images.to(device, non_blocking=True)

    outputs = model(images)
    logits = outputs["logits"]

    # Loss: soft CE for MixUp, standard CE otherwise
    if labels_soft is not None:
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(labels_soft * log_probs).sum(dim=1).mean()
        # Accuracy uses original hard labels
        labels_for_acc = batch.get("gt_label_orig", labels)
    else:
        loss_fn = LOSSES.build({"type": "CrossEntropyLoss"})
        loss = loss_fn(logits, labels)
        labels_for_acc = labels

    # Compute accuracy for logging
    pred = logits.argmax(dim=1)
    acc = (pred == labels_for_acc).float().mean()

    return {
        "loss": loss,
        "losses": {"ce": loss.detach()},
        "metrics": {"acc": acc.detach()},
    }


def forward_test(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run one eval forward pass."""
    images = batch["inputs"]
    if device is not None:
        images = images.to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(images)
    return {"logits": outputs["logits"].detach()}


def get_default_config() -> dict[str, Any]:
    return {
        "model": {
            "type": "resnet50",
            "backbone": {"type": "ResNet", "depth": 50},
            "head": {"type": "LinearClsHead", "num_classes": 1000},
        }
    }