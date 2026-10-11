"""Smooth L1 + IoU losses for detection — registered via LOSSES registry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import LOSSES


@LOSSES.register_module()
def SmoothL1Loss(beta: float = 1.0, reduction: str = "mean", **kwargs) -> nn.Module:
    """Smooth L1 loss for bounding box regression.

    Args:
        beta: Transition point between L1 and L2 regimes.
        reduction: 'mean' | 'sum' | 'none'.
    """
    return nn.SmoothL1Loss(beta=beta, reduction=reduction)


class _IoULoss(nn.Module):
    """Generalized IoU loss between two sets of boxes (N, 4) each."""

    def __init__(self, mode: str = "iou", reduction: str = "mean"):
        super().__init__()
        self.mode = mode  # 'iou' | 'giou'
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss.

        Boxes in format (x1, y1, x2, y2).
        """
        # Area of pred boxes
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        # Area of target boxes
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

        # Intersection
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        # Union
        union = pred_area + target_area - inter

        iou = inter / union.clamp(min=1e-6)

        if self.mode == "giou":
            # Enclosing box
            elt = torch.min(pred[:, :2], target[:, :2])
            erb = torch.max(pred[:, 2:], target[:, 2:])
            ewh = (erb - elt).clamp(min=0)
            enclose = ewh[:, 0] * ewh[:, 1]
            giou = iou - (enclose - union) / enclose.clamp(min=1e-6)
            loss = 1 - giou
        else:
            loss = 1 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@LOSSES.register_module()
def IoULoss(mode: str = "iou", reduction: str = "mean", **kwargs) -> nn.Module:
    """IoU / GIoU loss for bounding box regression."""
    if mode not in ("iou", "giou"):
        raise ValueError(f"unknown IoU mode: {mode!r}")
    return _IoULoss(mode=mode, reduction=reduction)


@LOSSES.register_module()
def GIoULoss(reduction: str = "mean", **kwargs) -> nn.Module:
    """Generalized IoU loss."""
    return _IoULoss(mode="giou", reduction=reduction)