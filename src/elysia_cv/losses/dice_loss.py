"""Dice loss for segmentation — registered via LOSSES registry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import LOSSES


class _DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation.

    Args:
        smooth: Laplace smoothing constant.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred: [B, C, H, W] logits.
            target: [B, H, W] class labels.

        Returns:
            Scalar loss.
        """
        num_classes = pred.size(1)
        pred_soft = F.softmax(pred, dim=1)

        # One-hot encode target
        target_onehot = F.one_hot(target.long(), num_classes=num_classes)  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Dice per class
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice  # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@LOSSES.register_module()
def DiceLoss(smooth: float = 1.0, reduction: str = "mean", **kwargs) -> nn.Module:
    """Dice loss for semantic segmentation."""
    return _DiceLoss(smooth=smooth, reduction=reduction)


@LOSSES.register_module()
def CombinedSegLoss(
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1.0,
    **kwargs,
) -> nn.Module:
    """Combined CrossEntropy + Dice loss for segmentation."""

    class _Combined(nn.Module):
        def __init__(self, ce_w: float, dice_w: float, smooth: float):
            super().__init__()
            self.ce_w = ce_w
            self.dice_w = dice_w
            self.dice = _DiceLoss(smooth=smooth, reduction="mean")

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            ce = F.cross_entropy(pred, target.long())
            dice = self.dice(pred, target)
            return self.ce_w * ce + self.dice_w * dice

    return _Combined(ce_weight, dice_weight, smooth)