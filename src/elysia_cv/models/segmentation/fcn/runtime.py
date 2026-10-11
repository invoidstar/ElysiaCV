"""FCN segmentation runtime — build / train / test bridge."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import BACKBONES, NECKS, HEADS, LOSSES


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build an FCN segmentation model from config.

    Layout: Backbone → IdentityNeck → SegHead
    """
    model_cfg = cfg.get("model", cfg)

    backbone_cfg = dict(model_cfg.get("backbone", {}))
    backbone_type = backbone_cfg.pop("type", "ResNet")
    backbone = BACKBONES.build({"type": backbone_type, **backbone_cfg})

    neck_cfg = dict(model_cfg.get("neck", {}))
    neck_type = neck_cfg.pop("type", "IdentityNeck")
    neck = NECKS.build({"type": neck_type, **neck_cfg})

    head_cfg = dict(model_cfg.get("head", {}))
    head_type = head_cfg.pop("type", "SegHead")
    head = HEADS.build({"type": head_type, **head_cfg})

    model = SegmentationModel(backbone=backbone, neck=neck, head=head)

    pretrained = model_cfg.get("pretrained_path")
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state, strict=False)

    return model


class SegmentationModel(nn.Module):
    """Backbone → Neck → Head segmentation model."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)


def forward_train(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """One training forward pass for segmentation.

    Expects batch keys:
      - ``inputs``: images [B, 3, H, W]
      - ``gt_semantic_seg``: [B, H, W] class labels

    Returns dict with ``loss`` and per-objective losses.
    """
    images = batch["inputs"]
    target = batch.get("gt_semantic_seg")
    if device is not None:
        images = images.to(device, non_blocking=True)
        if target is not None:
            target = target.to(device, non_blocking=True)

    # Forward
    logits = model(images)  # [B, C, h, w]

    # Upsample to target size if needed
    if target is not None and logits.shape[2:] != target.shape[1:]:
        logits = F.interpolate(
            logits, size=target.shape[1:], mode="bilinear", align_corners=False
        )

    # Combined CE + Dice loss
    loss_fn = LOSSES.build({"type": "CombinedSegLoss"})
    loss = loss_fn(logits, target)

    # Metric
    pred = logits.argmax(dim=1)
    acc = (pred == target).float().mean()

    return {
        "loss": loss,
        "losses": {"seg_loss": loss.detach()},
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
        logits = model(images)

    return {"logits": logits.detach()}


def get_default_config() -> dict[str, Any]:
    return {
        "model": {
            "type": "fcn",
            "backbone": {"type": "ResNet", "depth": 50, "out_indices": [4], "pool_last": False},
            "neck": {"type": "IdentityNeck"},
            "head": {"type": "SegHead", "num_classes": 150, "in_channels": 2048},
        }
    }