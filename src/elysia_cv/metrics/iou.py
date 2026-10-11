"""IoU / mIoU metrics — registered via METRICS registry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import METRICS


@METRICS.register_module()
class MeanIoU(nn.Module):
    """Mean Intersection over Union for semantic segmentation.

    Computes per-class IoU and averages over classes (ignoring background
    if specified).

    Args:
        num_classes: Total number of classes.
        ignore_index: Class index to ignore (-1 = none).
    """

    def __init__(self, num_classes: int = 150, ignore_index: int = -1, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate IoU statistics.

        Args:
            pred: [B, C, H, W] logits or [B, H, W] class indices.
            target: [B, H, W] class labels.
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        mask = (target != self.ignore_index) if self.ignore_index >= 0 else torch.ones_like(target, dtype=torch.bool)

        pred = pred[mask].cpu()
        target = target[mask].cpu()

        for cls in range(self.num_classes):
            p_mask = (pred == cls)
            t_mask = (target == cls)
            self.intersection[cls] += (p_mask & t_mask).sum().item()
            self.union[cls] += (p_mask | t_mask).sum().item()

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        """Return mIoU and per-class IoUs."""
        eps = 1e-6
        ious = (self.intersection + eps) / (self.union + eps)
        return {
            "mIoU": ious.mean().item(),
            "aAcc": self.intersection.sum().item() / max(self.union.sum().item(), 1),
        }


@METRICS.register_module()
class MeanAccuracy(nn.Module):
    """Mean pixel accuracy for segmentation."""

    def __init__(self, num_classes: int = 150, ignore_index: int = -1, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        self.correct += (pred == target).sum().item()
        self.total += target.numel()

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        return {"mAcc": self.correct / max(self.total, 1)}


@METRICS.register_module()
class MeanAP(nn.Module):
    """Placeholder mAP metric for object detection.

    Full COCO-style evaluation requires external packages (pycocotools).
    This stub computes a simple per-image average precision approximation.
    """

    def __init__(self, iou_thresholds: list[float] | None = None, **kwargs) -> None:
        super().__init__()
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.reset()

    def reset(self) -> None:
        self._ious: list[float] = []

    @torch.no_grad()
    def update(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> None:
        """Accumulate per-batch IoU statistics."""
        from elysia_cv.losses.smooth_l1 import _IoULoss
        iou_fn = _IoULoss(mode="iou", reduction="none")
        # Simple: loss → IoU = 1 - loss
        iou = 1 - iou_fn(pred_boxes, target_boxes).mean().item()
        self._ious.append(max(iou, 0.0))

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        if not self._ious:
            return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
        avg_iou = sum(self._ious) / len(self._ious)
        return {"mAP": avg_iou, "mAP_50": avg_iou, "mAP_75": avg_iou}