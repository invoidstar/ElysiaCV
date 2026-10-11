"""Faster R-CNN detection runtime — build / train / test bridge."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import BACKBONES, NECKS, HEADS, LOSSES


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a Faster R-CNN detection model from config.

    Layout: Backbone → FPN → DetHead
    """
    model_cfg = cfg.get("model", cfg)

    # Backbone
    backbone_cfg = dict(model_cfg.get("backbone", {}))
    backbone_type = backbone_cfg.pop("type", "ResNet")
    backbone = BACKBONES.build({"type": backbone_type, **backbone_cfg})

    # Neck (FPN)
    neck_cfg = dict(model_cfg.get("neck", {}))
    neck_type = neck_cfg.pop("type", "FPN")
    neck = NECKS.build({"type": neck_type, **neck_cfg})

    # Head
    head_cfg = dict(model_cfg.get("head", {}))
    head_type = head_cfg.pop("type", "DetHead")
    head = HEADS.build({"type": head_type, **head_cfg})

    model = DetectionModel(backbone=backbone, neck=neck, head=head)

    pretrained = model_cfg.get("pretrained_path")
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state, strict=False)

    return model


class DetectionModel(nn.Module):
    """Backbone → Neck → Head detection model."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        feats = self.backbone(x)
        # backbone returns tuple of multi-scale features
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        pyramid = self.neck(feats)
        return self.head(pyramid)


def forward_train(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """One training forward pass for detection.

    Expects batch keys:
      - ``inputs``: images [B, 3, H, W]
      - ``gt_bboxes``: list of [N_i, 4] bounding boxes
      - ``gt_labels``: list of [N_i] class labels

    Returns dict with ``loss`` and per-objective losses.
    """
    images = batch["inputs"]
    if device is not None:
        images = images.to(device, non_blocking=True)

    outputs = model(images)

    # Simplified detection loss: classification + regression on all
    # pyramid levels. In a real implementation this would use
    # matcher + assigner; here we use a simplified proxy loss.
    cls_scores = outputs["cls_score"]
    bbox_preds = outputs["bbox_pred"]

    # DetHead returns lists per FPN level; flatten all levels
    cls_preds = torch.cat([s.flatten(2).permute(0, 2, 1).reshape(-1, s.size(1))
                           for s in cls_scores], dim=0)  # [N, C]
    bbox_preds = torch.cat([b.flatten(2).permute(0, 2, 1).reshape(-1, 4)
                            for b in bbox_preds], dim=0)  # [N, 4]

    # Synthetic target: random assignments (demonstrates loss shape)
    if "gt_labels" in batch and isinstance(batch["gt_labels"], list):
        num_obj = sum(len(lbl) for lbl in batch["gt_labels"])
    else:
        num_obj = 0

    # Classification loss: assign random pseudo-labels for demo
    num_anchors = cls_preds.size(0)
    dummy_labels = torch.randint(0, cls_preds.size(1), (num_anchors,), device=cls_preds.device)
    cls_loss_fn = LOSSES.build({"type": "CrossEntropyLoss"})
    cls_loss = cls_loss_fn(cls_preds, dummy_labels)

    # Regression loss: simple L1 with random targets
    dummy_bboxes = torch.randn(num_anchors, 4, device=bbox_preds.device)
    reg_loss_fn = LOSSES.build({"type": "SmoothL1Loss"})
    reg_loss = reg_loss_fn(bbox_preds, dummy_bboxes)

    loss = cls_loss + reg_loss

    return {
        "loss": loss,
        "losses": {"cls_loss": cls_loss.detach(), "reg_loss": reg_loss.detach()},
        "metrics": {},
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

    return {
        "cls_score": [s.detach() for s in outputs["cls_score"]],
        "bbox_pred": [b.detach() for b in outputs["bbox_pred"]],
    }


def get_default_config() -> dict[str, Any]:
    return {
        "model": {
            "type": "faster_rcnn",
            "backbone": {"type": "ResNet", "depth": 50, "out_indices": [1, 2, 3, 4],
                         "pool_last": False},
            "neck": {"type": "FPN", "in_channels": [256, 512, 1024, 2048], "out_channels": 256},
            "head": {"type": "DetHead", "num_classes": 80},
        }
    }