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
    """COCO-style mean Average Precision for object detection.

    Pure PyTorch + NumPy implementation — no pycocotools dependency.

    Computes:
      - ``mAP``:     AP averaged over IoU thresholds [0.50:0.05:0.95]
      - ``mAP_50``:  AP at IoU=0.50
      - ``mAP_75``:  AP at IoU=0.75

    Usage::

        metric = MeanAP(num_classes=80)
        for batch in val_loader:
            preds = forward_test(model, batch)
            for i in range(len(batch)):
                metric.update(
                    pred_boxes=preds["pred_instances"][i]["bboxes"],
                    pred_scores=preds["pred_instances"][i]["scores"],
                    pred_labels=preds["pred_instances"][i]["labels"],
                    gt_boxes=batch["gt_bboxes"][i],
                    gt_labels=batch["gt_labels"][i],
                )
        results = metric.compute()
        metric.reset()
    """

    def __init__(
        self,
        num_classes: int = 80,
        iou_thresholds: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.iou_thresholds = (
            iou_thresholds
            if iou_thresholds is not None
            else [round(v, 2) for v in __import__("numpy").linspace(0.5, 0.95, 10)]
        )
        self.reset()

    # ------------------------------------------------------------------
    # Metric protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        # Per-image accumulators
        self._all_preds: list[dict] = []  # [{boxes, scores, labels}]
        self._all_gts: list[dict] = []     # [{boxes, labels, ignore}]

    @torch.no_grad()
    def update(
        self,
        pred_boxes: torch.Tensor,    # [P, 4]  xyxy
        pred_scores: torch.Tensor,   # [P]
        pred_labels: torch.Tensor,   # [P]
        gt_boxes: torch.Tensor,      # [G, 4]
        gt_labels: torch.Tensor,     # [G]
    ) -> None:
        """Accumulate one image's predictions and ground truths."""
        self._all_preds.append({
            "boxes": pred_boxes.detach().cpu(),
            "scores": pred_scores.detach().cpu(),
            "labels": pred_labels.detach().cpu(),
        })
        self._all_gts.append({
            "boxes": gt_boxes.detach().cpu(),
            "labels": gt_labels.detach().cpu(),
            "ignore": torch.zeros(len(gt_labels), dtype=torch.bool),
        })

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        """COCO-style mAP computation.

        Returns ``{"mAP": float, "mAP_50": float, "mAP_75": float}``.
        """
        if not self._all_gts:
            return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

        aps: list[float] = []
        for iou_thr in self.iou_thresholds:
            class_aps = []
            for cls_id in range(self.num_classes):
                ap = self._compute_ap_for_class(cls_id, iou_thr)
                if ap is not None:  # None = no GT for this class
                    class_aps.append(ap)
            if class_aps:
                aps.append(float(sum(class_aps)) / len(class_aps))
            else:
                aps.append(0.0)

        # mAP = mean over thresholds
        mAP = float(sum(aps)) / max(len(aps), 1)

        # mAP_50 = AP at 0.50
        idx_50 = 0  # first threshold is 0.50
        mAP_50 = aps[idx_50] if idx_50 < len(aps) else mAP

        # mAP_75 = AP at 0.75
        idx_75 = 5  # 0.50 + 5*0.05 = 0.75
        mAP_75 = aps[idx_75] if idx_75 < len(aps) else mAP

        # Store per-class PR data for later plotting
        self._last_pr_data: dict[str, list[Any]] | None = None

        return {"mAP": mAP, "mAP_50": mAP_50, "mAP_75": mAP_75}

    # ------------------------------------------------------------------
    # PR curve data extraction
    # ------------------------------------------------------------------

    def get_pr_data(self, iou_thr: float = 0.5) -> dict[str, list[float]]:
        """Extract per-class precision/recall curves for plotting.

        Must be called **after** ``compute()`` (which processes the
        accumulated data).  Returns a dict keyed by class ID with lists
        of ``[precision_values, recall_values]``.

        Args:
            iou_thr: IoU threshold for the PR curves (default 0.50).

        Returns:
            ``{cls_id: [precisions, recalls]}`` where each value is a
            list of 101 floats (interpolated to 101 recall levels).
        """
        import numpy as np

        from elysia_cv.metrics.iou import _box_iou

        result: dict[str, Any] = {}
        for cls_id in range(self.num_classes):
            # Collect all predictions for this class
            all_scores = []
            all_det_img = []
            all_det_boxes = []

            for img_idx, pred in enumerate(self._all_preds):
                mask = pred["labels"] == cls_id
                if not mask.any():
                    continue
                indices = mask.nonzero(as_tuple=True)[0]
                for j in indices:
                    all_scores.append(pred["scores"][j].item())
                    all_det_img.append(img_idx)
                    all_det_boxes.append(pred["boxes"][j].numpy())

            # Count GT instances and build matched list
            gt_boxes_per_img: list[list[np.ndarray]] = []
            gt_matched_per_img: list[list[bool]] = []
            for gt_item in self._all_gts:
                gt_mask = gt_item["labels"] == cls_id
                if gt_mask.any():
                    indices = gt_mask.nonzero(as_tuple=False)
                    boxes = [gt_item["boxes"][idx].numpy() for idx in indices.squeeze(-1)]
                else:
                    boxes = []
                gt_boxes_per_img.append(boxes)
                gt_matched_per_img.append([False] * len(boxes))

            npos = sum(len(b) for b in gt_boxes_per_img)
            if npos == 0 or not all_scores:
                continue

            # Sort by score descending
            scores_arr = np.array(all_scores)
            order = np.argsort(scores_arr)[::-1]

            tp = np.zeros(len(order), dtype=np.float32)
            fp = np.zeros(len(order), dtype=np.float32)

            for rank, ord_idx in enumerate(order):
                img_idx = all_det_img[ord_idx]
                det_box = all_det_boxes[ord_idx]
                gt_boxes = gt_boxes_per_img[img_idx]
                matched = gt_matched_per_img[img_idx]

                best_iou = 0.0
                best_j = -1
                for j, gt_box in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    iou_val = _box_iou_np(det_box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_j = j

                if best_iou >= iou_thr and best_j >= 0:
                    tp[rank] = 1.0
                    matched[best_j] = True
                else:
                    fp[rank] = 1.0

            # Cumulative
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float32).eps)
            rec = tp_cum / npos

            # 101-point interpolation
            recall_levels = np.linspace(0, 1, 101)
            prec_at_rec = np.zeros(101)
            for i, r in enumerate(recall_levels):
                mask_r = rec >= r
                prec_at_rec[i] = prec[mask_r].max() if mask_r.any() else 0.0

            result[str(cls_id)] = [prec_at_rec.tolist(), recall_levels.tolist()]

        return result

    # ------------------------------------------------------------------
    # Internal AP computation (11-point interpolation)
    # ------------------------------------------------------------------

    def _compute_ap_for_class(self, cls_id: int, iou_thr: float) -> float:
        """Compute AP for a single class at a single IoU threshold."""
        import numpy as np

        # Collect all predictions for this class
        all_scores: list[float] = []
        all_det_idx: list[int] = []   # which image
        all_det_local: list[int] = []  # which box within that image's preds

        for img_idx, pred in enumerate(self._all_preds):
            mask = pred["labels"] == cls_id
            if not mask.any():
                continue
            indices = mask.nonzero(as_tuple=True)[0]
            for j in indices:
                all_scores.append(pred["scores"][j].item())
                all_det_idx.append(img_idx)
                all_det_local.append(j.item())

        # Count total GT instances for this class
        gt_matched: list[dict[int, bool]] = [
            {j: False for j in range(len(gt["labels"]))
             if gt["labels"][j].item() == cls_id}
            for gt in self._all_gts
        ]
        npos = sum(len(m) for m in gt_matched)
        if npos == 0:
            return None  # exclude from averaging (no GT for this class)

        if not all_scores:
            return 0.0

        # Sort by score descending
        order = np.argsort(all_scores)[::-1]

        tp = np.zeros(len(order), dtype=np.float32)
        fp = np.zeros(len(order), dtype=np.float32)

        for rank, ord_idx in enumerate(order):
            img_idx = all_det_idx[ord_idx]
            det_j = all_det_local[ord_idx]

            det_box = self._all_preds[img_idx]["boxes"][det_j]  # [4]
            gt_boxes = self._all_gts[img_idx]["boxes"]
            gt_labels = self._all_gts[img_idx]["labels"]
            gt_cls_mask = gt_labels == cls_id

            if not gt_cls_mask.any():
                fp[rank] = 1.0
                continue

            gt_cls_boxes = gt_boxes[gt_cls_mask]  # [G_c, 4]
            gt_indices = gt_cls_mask.nonzero(as_tuple=True)[0].tolist()

            # Compute IoU with all GT boxes of this class in this image
            ious = _box_iou(det_box.unsqueeze(0), gt_cls_boxes)  # [1, G_c]
            best_iou, best_j = ious[0].max(dim=0)
            best_gt_idx = gt_indices[best_j.item()]

            if best_iou >= iou_thr and not gt_matched[img_idx].get(best_gt_idx, True):
                tp[rank] = 1.0
                gt_matched[img_idx][best_gt_idx] = True
            else:
                fp[rank] = 1.0

        # Compute precision-recall curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(npos, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

        # 101-point interpolation (COCO standard)
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            if np.any(rec >= t):
                ap += float(np.max(prec[rec >= t]))
        ap /= 101.0
        return ap


def _box_iou_np(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two single boxes (NumPy, xyxy format)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Args:
        boxes_a: [N, 4]
        boxes_b: [M, 4]

    Returns:
        [N, M] IoU matrix.
    """
    # Intersection
    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter_area

    return inter_area / (union + 1e-6)