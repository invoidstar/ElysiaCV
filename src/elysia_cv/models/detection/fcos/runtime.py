"""FCOS (Fully Convolutional One-Stage) detection runtime — build / train / test bridge.

Anchor-free single-stage detector with center-sampling target assignment,
per-FPN-level prediction, and NMS-based post-processing.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import BACKBONES, NECKS, HEADS, LOSSES

# ResNet-50 + FPN standard strides (P3 through P7)
STRIDES: list[int] = [8, 16, 32, 64, 128]


# ═══════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════

def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a detection model: Backbone → FPN → DetHead."""
    model_cfg = cfg.get("model", cfg)

    backbone_cfg = dict(model_cfg.get("backbone", {}))
    backbone_type = backbone_cfg.pop("type", "ResNet")
    backbone = BACKBONES.build({"type": backbone_type, **backbone_cfg})

    neck_cfg = dict(model_cfg.get("neck", {}))
    neck_type = neck_cfg.pop("type", "FPN")
    neck = NECKS.build({"type": neck_type, **neck_cfg})

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
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        pyramid = self.neck(feats)
        return self.head(pyramid)


# ═══════════════════════════════════════════════════════════════════
# Collate function — handles variable image sizes in a batch
# ═══════════════════════════════════════════════════════════════════

def _collate_detection_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate for detection: pad images to common max size.

    Images after keep_ratio resize may have different H×W; this pads
    them to the largest dimensions in the batch and stacks.
    """
    if not batch:
        return {}

    max_h = max(s["inputs"].shape[1] for s in batch)
    max_w = max(s["inputs"].shape[2] for s in batch)

    padded: list[torch.Tensor] = []
    for s in batch:
        img = s["inputs"]
        h, w = img.shape[1], img.shape[2]
        if h < max_h or w < max_w:
            img = F.pad(img, (0, max_w - w, 0, max_h - h))
        padded.append(img)

    return {
        "inputs": torch.stack(padded),                        # [B, C, max_H, max_W]
        "gt_bboxes": [s["gt_bboxes"] for s in batch],        # list of [N_i, 4]
        "gt_labels": [s["gt_labels"] for s in batch],         # list of [N_i]
        "image_id": [s.get("image_id", s.get("img_id", -1)) for s in batch],
        "pad_shape": (max_h, max_w),
    }


# ═══════════════════════════════════════════════════════════════════
# Target assignment — FCOS-style center sampling
# ═══════════════════════════════════════════════════════════════════

def _assign_targets(
    cls_scores: list[torch.Tensor],   # [B, C, H_l, W_l] each
    gt_bboxes: list[torch.Tensor],    # [[N_0,4], [N_1,4], ...] per image in batch
    gt_labels: list[torch.Tensor],    # [[N_0], ...]
    num_classes: int,
    strides: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Assign classification and regression targets for each FPN level.

    Returns:
        cls_targets: per-level [B, H_l, W_l] — 0 for bg, 1..C for classes
        reg_targets: per-level [B, 4, H_l, W_l] — (l, t, r, b) distances
        pos_masks:   per-level [B, H_l, W_l] — True where a box was matched
    """
    B = cls_scores[0].shape[0]
    device = cls_scores[0].device

    all_cls_targets: list[torch.Tensor] = []
    all_reg_targets: list[torch.Tensor] = []
    all_pos_masks:   list[torch.Tensor] = []

    for lvl, (scores, stride) in enumerate(zip(cls_scores, strides)):
        _, _, H, W = scores.shape
        cls_tar = torch.zeros(B, H, W, dtype=torch.long, device=device)
        reg_tar = torch.zeros(B, 4, H, W, dtype=torch.float32, device=device)
        pos_msk = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        for b in range(B):
            bboxes = gt_bboxes[b]   # [N_b, 4]  xyxy
            labels = gt_labels[b]   # [N_b]

            if bboxes.numel() == 0:
                continue

            # For each position (h, w), compute image center: (w+0.5)*s, (h+0.5)*s
            ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride  # [H]
            xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride  # [W]
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

            # For each GT box, check which centers fall inside
            for gt_box, gt_label in zip(bboxes, labels):
                x1, y1, x2, y2 = gt_box
                inside = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)

                if not inside.any():
                    # If no center falls inside (small box), assign the nearest position
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    ci = (cy / stride).long().clamp(0, H - 1)
                    cj = (cx / stride).long().clamp(0, W - 1)
                    inside[ci, cj] = True

                # Handle conflicts: if a position was already assigned, keep the
                # one with smaller GT area (more precise)
                current_area = (x2 - x1) * (y2 - y1)
                for h_idx, w_idx in zip(*torch.where(inside)):
                    if pos_msk[b, h_idx, w_idx]:
                        # Check if current box is smaller → replace
                        old_label = cls_tar[b, h_idx, w_idx] - 1
                        old_box = None
                        for old_gt, old_lbl in zip(bboxes, labels):
                            if old_lbl == old_label:
                                old_box = old_gt
                                break
                        if old_box is not None:
                            old_area = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
                            if current_area >= old_area:
                                continue  # keep previous assignment

                    cls_tar[b, h_idx, w_idx] = gt_label + 1  # 0=bg, 1..C=fg
                    pos_msk[b, h_idx, w_idx] = True

                    # Regression: l, t, r, b
                    reg_tar[b, 0, h_idx, w_idx] = grid_x[h_idx, w_idx] - x1  # l
                    reg_tar[b, 1, h_idx, w_idx] = grid_y[h_idx, w_idx] - y1  # t
                    reg_tar[b, 2, h_idx, w_idx] = x2 - grid_x[h_idx, w_idx]  # r
                    reg_tar[b, 3, h_idx, w_idx] = y2 - grid_y[h_idx, w_idx]  # b

        all_cls_targets.append(cls_tar)
        all_reg_targets.append(reg_tar)
        all_pos_masks.append(pos_msk)

    return all_cls_targets, all_reg_targets, all_pos_masks


# ═══════════════════════════════════════════════════════════════════
# Loss computation
# ═══════════════════════════════════════════════════════════════════

def _compute_cls_loss(
    cls_scores: list[torch.Tensor],
    cls_targets: list[torch.Tensor],
    pos_masks: list[torch.Tensor],
    num_classes: int,
    bg_weight: float = 1.0,
) -> torch.Tensor:
    """Classification loss: CE with positive/negative sampling.

    Target encoding: 0=bg, 1..C=fg class.  Converted to 0..C-1 for the
    C-class head (bg excluded from loss via ignore_index).
    """
    cls_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    losses = []
    for scores, targets, pos in zip(cls_scores, cls_targets, pos_masks):
        B, C, H, W = scores.shape

        scores_flat = scores.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
        targets_flat = targets.reshape(-1)                          # [B*H*W]  values 0..C
        pos_flat = pos.reshape(-1)                                  # [B*H*W]

        # Convert targets: 0=bg→ignore, 1..C=fg→0..C-1
        targets_adjusted = targets_flat.clone()
        targets_adjusted[targets_flat == 0] = -100
        targets_adjusted[targets_flat > 0] = targets_adjusted[targets_flat > 0] - 1

        n_pos = pos_flat.sum().item()
        if n_pos == 0:
            continue  # no positive samples in this level

        n_neg = min(n_pos, int((~pos_flat).sum().item()))

        pos_idx = torch.where(pos_flat)[0]
        neg_pool = torch.where(~pos_flat)[0]
        if len(neg_pool) > 0:
            neg_idx = neg_pool[torch.randperm(len(neg_pool), device=neg_pool.device)[:n_neg]]
            all_idx = torch.cat([pos_idx, neg_idx])
        else:
            all_idx = pos_idx

        loss = cls_loss_fn(scores_flat[all_idx], targets_adjusted[all_idx])
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=cls_scores[0].device, requires_grad=True)
    return torch.stack(losses).mean()


def _compute_reg_loss(
    bbox_preds: list[torch.Tensor],
    reg_targets: list[torch.Tensor],
    pos_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Regression loss: SmoothL1 on positive samples only."""
    reg_loss_fn = LOSSES.build({"type": "SmoothL1Loss", "beta": 1.0 / 9.0})

    total_loss = 0.0
    total_pos = 0

    for preds, targets, pos in zip(bbox_preds, reg_targets, pos_masks):
        B, _, H, W = preds.shape
        preds_flat = preds.permute(0, 2, 3, 1).reshape(-1, 4)    # [B*H*W, 4]
        targets_flat = targets.permute(0, 2, 3, 1).reshape(-1, 4)  # [B*H*W, 4]
        pos_flat = pos.reshape(-1)                                   # [B*H*W]

        if pos_flat.sum().item() == 0:
            continue

        loss = reg_loss_fn(preds_flat[pos_flat], targets_flat[pos_flat])
        total_loss += loss * pos_flat.sum().item()
        total_pos += pos_flat.sum().item()

    if total_pos == 0:
        return torch.tensor(0.0, device=bbox_preds[0].device, requires_grad=True)
    return total_loss / max(total_pos, 1)


# ═══════════════════════════════════════════════════════════════════
# Training / testing interface
# ═══════════════════════════════════════════════════════════════════

def forward_train(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """One training forward pass with real target assignment.

    Args:
        model: DetectionModel instance.
        batch: Collated batch dict with:
          - ``inputs``: [B, 3, H, W]
          - ``gt_bboxes``: list of [N_i, 4] tensors (xyxy)
          - ``gt_labels``:  list of [N_i] tensors (0-based class)

    Returns:
        dict with ``loss`` and ``losses``.
    """
    images = batch["inputs"]
    gt_bboxes = batch.get("gt_bboxes", [])
    gt_labels = batch.get("gt_labels", [])

    if device is not None:
        images = images.to(device, non_blocking=True)
        gt_bboxes = [b.to(device) if isinstance(b, torch.Tensor) else b
                      for b in gt_bboxes]
        gt_labels = [l.to(device) if isinstance(l, torch.Tensor) else l
                      for l in gt_labels]

    # 1. Forward through backbone + FPN + head
    outputs = model(images)
    cls_scores: list[torch.Tensor] = outputs["cls_score"]  # [B, C, H_l, W_l]
    bbox_preds: list[torch.Tensor] = outputs["bbox_pred"]  # [B, 4, H_l, W_l]

    num_classes = cls_scores[0].shape[1]

    # 2. Build strides list (match number of FPN levels)
    strides = STRIDES[-len(cls_scores):]  # in case num_outs < 5

    # 3. Assign targets per FPN level
    cls_targets, reg_targets, pos_masks = _assign_targets(
        cls_scores, gt_bboxes, gt_labels, num_classes, strides,
    )

    # 4. Compute losses
    cls_loss = _compute_cls_loss(cls_scores, cls_targets, pos_masks, num_classes)
    reg_loss = _compute_reg_loss(bbox_preds, reg_targets, pos_masks)

    loss = cls_loss + reg_loss

    return {
        "loss": loss,
        "losses": {
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
        },
        "metrics": {},
    }


def _decode_detections(
    cls_scores: list[torch.Tensor],
    bbox_preds: list[torch.Tensor],
    strides: list[int],
    score_thr: float = 0.05,
    nms_thr: float = 0.5,
    max_per_img: int = 100,
) -> list[dict[str, torch.Tensor]]:
    """Decode FCOS-style outputs into per-image detection results.

    Returns:
        List of dicts (one per image): ``{"bboxes": [K,4], "scores": [K], "labels": [K]}``.
    """
    B = cls_scores[0].shape[0]
    device = cls_scores[0].device
    num_classes = cls_scores[0].shape[1]

    all_bboxes: list[list[torch.Tensor]] = [[] for _ in range(B)]
    all_scores: list[list[torch.Tensor]] = [[] for _ in range(B)]
    all_labels: list[list[torch.Tensor]] = [[] for _ in range(B)]

    for lvl, (scores, preds, stride) in enumerate(zip(cls_scores, bbox_preds, strides)):
        _, _, H, W = scores.shape  # [B, C, H, W]

        # Map positions to image coordinates
        ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

        # Per-position scores: take max across classes, apply sigmoid
        scores_lvl = scores.sigmoid()  # [B, C, H, W]
        max_scores, max_labels = scores_lvl.max(dim=1)  # [B, H, W]

        # Per-position bbox decode: l,t,r,b → x1,y1,x2,y2
        # bbox_preds: [B, 4, H, W] → (l, t, r, b)
        l = preds[:, 0, :, :]  # [B, H, W]
        t = preds[:, 1, :, :]
        r = preds[:, 2, :, :]
        b = preds[:, 3, :, :]

        # Gather per-image
        for bi in range(B):
            scores_bi = max_scores[bi]     # [H, W]
            labels_bi = max_labels[bi]     # [H, W]

            # Filter by score threshold
            keep = scores_bi > score_thr
            if not keep.any():
                continue

            sc = scores_bi[keep]
            lb = labels_bi[keep]
            cx = grid_x[keep]  # center x in image coords
            cy = grid_y[keep]
            left = l[bi][keep]
            top = t[bi][keep]
            right = r[bi][keep]
            bottom = b[bi][keep]

            boxes = torch.stack([
                cx - left, cy - top,
                cx + right, cy + bottom,
            ], dim=1)  # [K, 4]  xyxy

            all_bboxes[bi].append(boxes)
            all_scores[bi].append(sc)
            all_labels[bi].append(lb)

    # Gather per-image results + NMS
    results: list[dict[str, torch.Tensor]] = []
    for bi in range(B):
        if not all_bboxes[bi]:
            results.append({
                "bboxes": torch.zeros(0, 4, device=device),
                "scores": torch.zeros(0, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device),
            })
            continue

        boxes = torch.cat(all_bboxes[bi], dim=0)    # [N, 4]
        scores = torch.cat(all_scores[bi], dim=0)    # [N]
        labels = torch.cat(all_labels[bi], dim=0)    # [N]

        # Keep top-K before NMS
        if len(scores) > 1000:
            topk = scores.topk(1000).indices
            boxes, scores, labels = boxes[topk], scores[topk], labels[topk]

        # Per-class NMS
        keep_mask = torch.zeros(len(scores), dtype=torch.bool, device=device)
        for cls in labels.unique():
            cls_mask = labels == cls
            cls_idx = cls_mask.nonzero(as_tuple=True)[0]
            cls_boxes = boxes[cls_idx]
            cls_scores = scores[cls_idx]

            keep_idx = _nms(cls_boxes, cls_scores, nms_thr, max_per_img)
            keep_mask[cls_idx[keep_idx]] = True

        results.append({
            "bboxes": boxes[keep_mask],
            "scores": scores[keep_mask],
            "labels": labels[keep_mask],
        })

    return results


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float, max_dets: int) -> torch.Tensor:
    """Pure-PyTorch NMS. Returns indices to keep."""
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # Sort by score descending
    order = scores.argsort(descending=True)
    keep: list[int] = []

    while order.numel() > 0 and len(keep) < max_dets:
        idx = order[0].item()
        keep.append(idx)
        if order.numel() == 1:
            break

        # IoU of highest-scoring box vs the rest
        box_a = boxes[idx].unsqueeze(0)  # [1, 4]
        box_b = boxes[order[1:]]          # [N-1, 4]

        # Compute IoU
        inter_x1 = torch.max(box_a[:, 0], box_b[:, 0])
        inter_y1 = torch.max(box_a[:, 1], box_b[:, 1])
        inter_x2 = torch.min(box_a[:, 2], box_b[:, 2])
        inter_y2 = torch.min(box_a[:, 3], box_b[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        iou = inter_area / (area_a + area_b - inter_area + 1e-6)

        rem = iou <= iou_thr
        order = order[1:][rem]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def forward_test(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run one eval forward pass with decoded detections."""
    images = batch["inputs"]
    if device is not None:
        images = images.to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(images)

    cls_scores = outputs["cls_score"]
    bbox_preds = outputs["bbox_pred"]
    strides = STRIDES[-len(cls_scores):]

    decoded = _decode_detections(cls_scores, bbox_preds, strides)

    return {
        "cls_score": [s.detach() for s in cls_scores],
        "bbox_pred": [b.detach() for b in bbox_preds],
        "pred_instances": decoded,  # list of {bboxes, scores, labels}
    }


def get_default_config() -> dict[str, Any]:
    return {
        "model": {
            "type": "fcos",
            "backbone": {"type": "ResNet", "depth": 50, "out_indices": [1, 2, 3, 4],
                         "pool_last": False},
            "neck": {"type": "FPN", "in_channels": [256, 512, 1024, 2048], "out_channels": 256},
            "head": {"type": "DetHead", "num_classes": 80},
        }
    }