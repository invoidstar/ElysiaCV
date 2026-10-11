"""Accuracy metrics — registered via METRICS registry."""

from __future__ import annotations

import torch
import torch.nn as nn

from elysia_cv.core.registry import METRICS


@METRICS.register_module()
class Accuracy(nn.Module):
    """Top-1 accuracy metric.

    Usage::

        acc = Accuracy()
        acc.update(pred, target)  # accumulate
        result = acc.compute()    # average over all updates
        acc.reset()
    """

    def __init__(self, topk: int = 1, **kwargs) -> None:
        super().__init__()
        self.topk = topk
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate batch predictions."""
        maxk = min(self.topk, pred.size(1))
        _, indices = pred.topk(maxk, dim=1)
        correct = indices.eq(target.view(-1, 1).expand_as(indices))
        self.correct += correct.sum().item()
        self.total += target.size(0)

    @torch.no_grad()
    def compute(self) -> float:
        """Return average accuracy (0-1)."""
        return self.correct / max(self.total, 1)


@METRICS.register_module()
class Top5Accuracy(Accuracy):
    """Top-5 accuracy metric."""

    def __init__(self, **kwargs) -> None:
        super().__init__(topk=5, **kwargs)