"""TaskAlignedAssigner — dynamic label assignment for YOLOv8/v10/v11/v12/v26.

Given cls scores and predicted bboxes, computes an alignment score
(alpha-weighted cls confidence × beta-weighted IoU), selects top-k
anchors, and resolves conflicts by keeping the highest alignment.

Reference
---------
Task-aligned One-stage Object Detection (Feng et al., CVPR 2021)
YOLOv8 / YOLOv11 implementation in ultralytics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _make_grid(
    h: int, w: int, stride: torch.Tensor | int, dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (grid_x, grid_y) in image coordinates (shape: [1, h*w, 1])."""
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )
    return (gx.reshape(1, -1, 1) + 0.5) * stride, (gy.reshape(1, -1, 1) + 0.5) * stride


def _bbox_iou(
    b1: torch.Tensor, b2: torch.Tensor,
) -> torch.Tensor:
    """IoU of two sets of xyxy boxes.

    Parameters
    ----------
    b1 : [N, 4]  or  [B, M, 4]
    b2 : [M, 4]  or  [B, M, 4]
    """
    # Inter
    lt = torch.max(b1[..., :2].unsqueeze(2), b2[..., :2].unsqueeze(1))
    rb = torch.min(b1[..., 2:].unsqueeze(2), b2[..., 2:].unsqueeze(1))
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])  # [B,N]
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])  # [B,M]
    union = area1.unsqueeze(2) + area2.unsqueeze(1) - inter
    return inter / (union + 1e-16)


class TaskAlignedAssigner:
    """Aligns anchors to ground-truth boxes using a joint cls×IoU metric.

    Parameters
    ----------
    topk : int
        Number of top-aligned anchors to select per GT.
    nc : int
        Number of classes.
    alpha : float
        Exponent for classification score (default 1.0).
    beta : float
        Exponent for IoU score (default 6.0).
    """

    def __init__(self, topk: int = 13, nc: int = 80, alpha: float = 1.0, beta: float = 6.0):
        self.topk = topk
        self.nc = nc
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def __call__(
        self,
        cls_preds: torch.Tensor,        # [B, num_anchors, nc]
        bbox_preds: torch.Tensor,       # [B, num_anchors, 4]  (xyxy)
        anchor_points: torch.Tensor,    # [B, num_anchors, 2]  (cx, cy in image coords)
        gt_labels: torch.Tensor,        # [B, max_num_gt, 1]
        gt_bboxes: torch.Tensor,        # [B, max_num_gt, 4]  (xyxy)
        mask_gt: torch.Tensor,          # [B, max_num_gt] bool — True where a GT exists
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (target_labels, target_bboxes, target_scores, fg_mask)."""
        B, na, _ = cls_preds.shape
        device = cls_preds.device
        dtype = cls_preds.dtype

        cls_preds = cls_preds.sigmoid()

        target_labels_l = torch.full((B, na), self.nc, dtype=torch.long, device=device)
        target_bboxes_l = torch.zeros(B, na, 4, dtype=dtype, device=device)
        target_scores_l = torch.zeros(B, na, dtype=dtype, device=device)
        fg_mask = torch.zeros(B, na, dtype=torch.bool, device=device)

        # Process each sample individually — avoids per‑batch IoU broadcast bugs
        for b in range(B):
            ngt = mask_gt[b].sum().item()
            if ngt == 0:
                continue

            c_b = cls_preds[b:b+1]          # [1, na, nc]
            bx_b = bbox_preds[b:b+1]         # [1, na, 4]
            g_b = gt_bboxes[b:b+1, :int(ngt)]   # [1, ngt, 4]
            lab_b = gt_labels[b:b+1, :int(ngt), 0]  # [1, ngt]

            # IoU: [1, na, ngt]
            ious = _bbox_iou(bx_b, g_b)     # [1, na, ngt]

            # Grab cls scores for GT classes only
            gt_cls = c_b[0, :, lab_b[0].long()]  # [na, ngt]

            # Alignment metric: cls^α × IoU^β
            align = gt_cls.pow(self.alpha) * ious.squeeze(0).pow(self.beta)  # [na, ngt]

            # Best GT per anchor
            scores_best, idx_best = align.max(dim=-1)  # [na], [na]

            # Top-k anchors per GT
            align_t = align.t()   # [ngt, na]
            topk = min(self.topk, na)
            _, topk_idx = align_t.topk(topk, dim=-1)  # [ngt, topk]

            # Mask: anchor ∈ topk for any GT
            topk_mask = torch.zeros(na, dtype=torch.bool, device=device)
            for t in range(int(ngt)):
                topk_mask[topk_idx[t]] = True

            fg_mask[b] = topk_mask

            if fg_mask[b].any():
                target_labels_l[b][fg_mask[b]] = lab_b[0, idx_best[fg_mask[b]]]
                target_scores_l[b][fg_mask[b]] = scores_best[fg_mask[b]]
                target_bboxes_l[b][fg_mask[b]] = g_b[0, idx_best[fg_mask[b]]]

        return target_labels_l, target_bboxes_l, target_scores_l, fg_mask