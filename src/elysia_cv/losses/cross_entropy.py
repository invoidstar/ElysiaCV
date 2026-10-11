"""Cross-entropy loss — registered via LOSSES registry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import LOSSES


@LOSSES.register_module()
def CrossEntropyLoss(
    weight: list[float] | None = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """Cross-entropy loss with optional label smoothing.

    Returns a callable module (``nn.Module`` or ``nn.CrossEntropyLoss``).
    """
    weight_tensor = torch.tensor(weight) if weight is not None else None
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing, reduction=reduction)


@LOSSES.register_module()
def FocalLoss(
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """Focal Loss for class-imbalanced classification."""

    class _FocalLoss(nn.Module):
        def __init__(self, alpha: float, gamma: float, reduction: str):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ce = F.cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-ce)
            loss = self.alpha * (1 - pt) ** self.gamma * ce
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

    return _FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)