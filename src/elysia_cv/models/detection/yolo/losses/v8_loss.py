"""YOLOv8/v11 loss — CIoU box regression + DFL + BCE classification.

The loss operates on per‑anchor predictions produced by the decoupled head:
    cls_scores  — list of [B, num_classes, H_l, W_l] tensors
    reg_preds   — list of [B, reg_max*4, H_l, W_l] tensors

Each (H_l, W_l) grid cell contributes one anchor point.

Reference
---------
YOLOv8 / YOLOv11 (ultralytics)
Distribution Focal Loss (DFL), Li et al., 2020
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _make_anchors(
    feats: list[torch.Tensor],
    strides: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (anchor_points, stride_tensor) for all feature levels.

    anchor_points: [B, total_na, 2]  — (cx, cy) in input image coords
    stride_tensor: [B, total_na, 1]
    """
    anchors, st = [], []
    for f, s in zip(feats, strides):
        _, _, h, w = f.shape
        gy, gx = torch.meshgrid(
            torch.arange(h, dtype=dtype, device=device),
            torch.arange(w, dtype=dtype, device=device),
            indexing="ij",
        )
        anchors.append(
            torch.stack([(gx + 0.5) * s, (gy + 0.5) * s], dim=-1).reshape(1, -1, 2)
        )
        st.append(torch.full((1, h * w, 1), s, dtype=dtype, device=device))
    return torch.cat(anchors, dim=1), torch.cat(st, dim=1)


def _dist_to_bbox(
    points: torch.Tensor, dist: torch.Tensor, stride: torch.Tensor
) -> torch.Tensor:
    """Convert distribution (DFL bins) to xyxy boxes.

    Parameters
    ----------
    points : [B, na, 2]
    dist   : [B, na, 4]   —  (l, t, r, b)  in pixels (relative to stride)
    stride : [B, na, 1]
    """
    lt = points - dist[..., :2] * stride
    rb = points + dist[..., 2:] * stride
    return torch.cat([lt, rb], dim=-1)


def _bbox_iou(b1: torch.Tensor, b2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU of xyxy boxes."""
    lt = torch.max(b1[..., :2], b2[..., :2])
    rb = torch.min(b1[..., 2:], b2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    return inter / (area1 + area2 - inter + eps)


# ---------------------------------------------------------------------------
# CIoU loss
# ---------------------------------------------------------------------------

def _ciou_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Complete IoU loss — scalar per box pair."""
    iou = _bbox_iou(pred, target, eps)

    # Enclosures
    lt = torch.min(pred[..., :2], target[..., :2])
    rb = torch.max(pred[..., 2:], target[..., 2:])
    wh_enc = (rb - lt).clamp(min=eps)
    enc_diag = wh_enc[..., 0] ** 2 + wh_enc[..., 1] ** 2

    # Center distance
    c_pred = (pred[..., :2] + pred[..., 2:]) / 2
    c_tgt = (target[..., :2] + target[..., 2:]) / 2
    rho2 = (c_pred[..., 0] - c_tgt[..., 0]) ** 2 + (c_pred[..., 1] - c_tgt[..., 1]) ** 2

    # Aspect ratio penalty
    w_pred, h_pred = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
    w_tgt, h_tgt = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
    v = (4 / (math.pi ** 2)) * (torch.atan(w_tgt / (h_tgt + eps)) - torch.atan(w_pred / (h_pred + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = 1 - iou + rho2 / (enc_diag + eps) + alpha * v
    return ciou


# ---------------------------------------------------------------------------
# Distribution Focal Loss
# ---------------------------------------------------------------------------

def _dfl_loss(
    pred_dists: torch.Tensor,        # [N, reg_max * 4]
    targets: torch.Tensor,           # [N, 4]  —  l,t,r,b distances (pixels)
    strides: torch.Tensor,           # [N, 1]
    reg_max: int = 16,
) -> torch.Tensor:
    """Distribution Focal Loss — per-coordinate.

    Parameters
    ----------
    pred_dists : [N_fg, reg_max * 4]
        Raw logits (not softmaxed).
    targets : [N_fg, 4]
        Ground‑truth l,t,r,b in image pixels.
    """
    if pred_dists.numel() == 0:
        return pred_dists.sum() * 0

    N = pred_dists.shape[0]
    pred_dists = pred_dists.reshape(N, 4, reg_max)         # [N, 4, reg_max]

    # Scale targets to reg_max bins and clamp
    tgt = (targets / strides).clamp(0, reg_max - 1 - 1e-3)      # [N, 4]
    tgt_lo = tgt.long()
    tgt_hi = tgt_lo + 1
    w_hi = (tgt - tgt_lo.float()).clamp(0, 1)
    w_lo = 1 - w_hi

    # Gather logits at left/right bins
    pred_lo = pred_dists.gather(-1, tgt_lo.unsqueeze(-1).clamp(0, reg_max - 1)).squeeze(-1)  # [N, 4]
    pred_hi = pred_dists.gather(-1, tgt_hi.unsqueeze(-1).clamp(0, reg_max - 1)).squeeze(-1)

    loss = (
        F.cross_entropy(pred_dists.reshape(-1, reg_max), tgt_lo.clamp(0, reg_max - 1).reshape(-1), reduction="none")
        .reshape(N, 4).mean(1)
    )
    return loss.mean()


# ---------------------------------------------------------------------------
# YOLO Loss (full)
# ---------------------------------------------------------------------------

class YOLOv8Loss(nn.Module):
    """YOLOv8/v11 detection loss — TAL + CIoU + DFL + BCE.

    Parameters
    ----------
    num_classes : int
    reg_max : int, default 16
    tal_topk : int, default 13
    tal_alpha : float, default 1.0
    tal_beta : float, default 6.0
    label_smoothing : float, default 0.0
    strides : list[int], default [8, 16, 32]
    loss_weights : dict | None
    """

    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        tal_topk: int = 13,
        tal_alpha: float = 1.0,
        tal_beta: float = 6.0,
        label_smoothing: float = 0.0,
        strides: list[int] | None = None,
        loss_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.label_smoothing = label_smoothing
        self.strides = strides or [8, 16, 32]

        from .task_aligned_assigner import TaskAlignedAssigner
        self.assigner = TaskAlignedAssigner(topk=tal_topk, nc=num_classes, alpha=tal_alpha, beta=tal_beta)
        self.loss_weights = loss_weights or {"cls": 0.5, "box": 7.5, "dfl": 1.5}

        # BCE reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        cls_scores: list[torch.Tensor],     # list of [B, nc, H, W]
        reg_preds: list[torch.Tensor],     # list of [B, 4*reg_max, H, W]
        gt_bboxes: list[torch.Tensor],      # list of [num_gt_i, 4] (xyxy, on the target device)
        gt_labels: list[torch.Tensor],      # list of [num_gt_i]
        images: torch.Tensor | None = None,  # [B, 3, H, W] for stride-aware anchor gen
    ) -> dict[str, torch.Tensor]:
        """Compute losses.

        Returns  ``{"loss": total, "losses": {key: ...}, "metrics": {"iou": ...}}``.
        """
        B = len(cls_scores[0])
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype

        # ---- Flatten predictions into per-anchor form ----
        anchors: list[torch.Tensor] = []
        cls_list: list[torch.Tensor] = []
        reg_list: list[torch.Tensor] = []
        stride_list: list[torch.Tensor] = []

        for lvl, (c, r, s) in enumerate(zip(cls_scores, reg_preds, self.strides)):
            _, _, h, w = c.shape
            na_l = h * w

            # Anchor points
            gy, gx = torch.meshgrid(
                torch.arange(h, dtype=dtype, device=device),
                torch.arange(w, dtype=dtype, device=device),
                indexing="ij",
            )
            ap = torch.stack([(gx + 0.5) * s, (gy + 0.5) * s], dim=-1).reshape(1, na_l, 2).expand(B, -1, -1)
            anchors.append(ap)

            cls_list.append(c.reshape(B, self.nc, na_l).permute(0, 2, 1))          # [B, na, nc]
            reg_list.append(r.reshape(B, self.reg_max * 4, na_l).permute(0, 2, 1)) # [B, na, reg_max*4]
            stride_list.append(torch.full((1, na_l, 1), s, dtype=dtype, device=device).expand(B, -1, -1))

        anchor_points = torch.cat(anchors, dim=1)       # [B, total_na, 2]
        cls_all = torch.cat(cls_list, dim=1)              # [B, total_na, nc]
        reg_all = torch.cat(reg_list, dim=1)              # [B, total_na, 4*reg_max]
        stride_all = torch.cat(stride_list, dim=1)        # [B, total_na, 1]

        # ---- Pad GT boxes to a uniform tensor ----
        max_gt = max(len(g) for g in gt_labels)
        gt_labels_t = torch.full((B, max_gt, 1), self.nc, dtype=torch.long, device=device)
        gt_bboxes_t = torch.zeros(B, max_gt, 4, dtype=dtype, device=device)
        mask_gt = torch.zeros(B, max_gt, dtype=torch.bool, device=device)

        for b in range(B):
            n = len(gt_labels[b])
            if n == 0:
                continue
            gt_labels_t[b, :n, 0] = gt_labels[b][:n].long()
            gt_bboxes_t[b, :n, :] = gt_bboxes[b][:n]
            mask_gt[b, :n] = True

        # ---- Decode predicted boxes (for TAL) ----
        # Convert distribution to l,t,r,b then to xyxy
        pred_dist_4 = reg_all.view(B, -1, 4, self.reg_max)  # [B, na, 4, reg_max]
        proj = torch.arange(self.reg_max, dtype=dtype, device=device)
        # Softmax projection
        pred_lt = (pred_dist_4.softmax(-1) @ proj)             # [B, na, 4]  (l,t,r,b in bins)
        # to image pixels
        pred_bboxes = _dist_to_bbox(anchor_points, pred_lt * stride_all, stride_all)  # [B, na, 4] xyxy

        # ---- TAL assigner ----
        target_labels_l, target_bboxes_l, target_scores_l, fg_mask = self.assigner(
            cls_all, pred_bboxes, anchor_points, gt_labels_t, gt_bboxes_t, mask_gt
        )

        # ---- Classification loss (BCE) ----
        target_cls = F.one_hot(target_labels_l.clamp(0, self.nc - 1), self.nc).float()  # [B, na, nc]
        if self.label_smoothing > 0:
            target_cls = target_cls * (1 - self.label_smoothing) + self.label_smoothing / self.nc
        # Weight foreground samples more
        weight_cls = (fg_mask.unsqueeze(-1).float() * target_scores_l.unsqueeze(-1)).detach()
        cls_loss = self.bce(cls_all, target_cls)
        cls_loss = (cls_loss * weight_cls).sum() / max(weight_cls.sum(), 1)

        # ---- Box regression loss (CIoU) ----
        if fg_mask.any():
            fg_pred = pred_bboxes[fg_mask]
            fg_tgt = target_bboxes_l[fg_mask]
            iou = _bbox_iou(fg_pred, fg_tgt).detach()
            box_loss = _ciou_loss(fg_pred, fg_tgt).mean()
        else:
            box_loss = torch.tensor(0.0, device=device, dtype=dtype)
            iou = torch.tensor(0.0, device=device, dtype=dtype)

        # ---- DFL loss ----
        if fg_mask.any():
            # Get l,t,r,b for foreground anchors
            valid_mask = fg_mask.float().unsqueeze(-1)           # [B, na, 1]
            pred_lt_fg = (pred_dist_4.softmax(-1) @ proj)         # [B, na, 4]
            # GT l,t,r,b = target xyxy → distance from anchor point
            ap_fg = anchor_points
            tgt_lt = torch.cat([
                ap_fg[..., :2] - target_bboxes_l[..., :2],        # l,t
                target_bboxes_l[..., 2:] - ap_fg[..., :2],         # r,b
            ], dim=-1).clamp(min=0)

            dfl = _dfl_loss(reg_all[fg_mask], tgt_lt[fg_mask], stride_all[fg_mask], self.reg_max)
        else:
            dfl = torch.tensor(0.0, device=device, dtype=dtype)

        # ---- Weighted sum ----
        w_cls = self.loss_weights.get("cls", 0.5)
        w_box = self.loss_weights.get("box", 7.5)
        w_dfl = self.loss_weights.get("dfl", 1.5)

        total_loss = w_cls * cls_loss + w_box * box_loss + w_dfl * dfl

        return {
            "loss": total_loss,
            "losses": {"cls_loss": cls_loss, "box_loss": box_loss, "dfl_loss": dfl},
            "metrics": {"iou": iou.mean() if fg_mask.any() else iou, "fg_ratio": fg_mask.float().mean()},
        }