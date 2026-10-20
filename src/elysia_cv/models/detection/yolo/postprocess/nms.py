"""YOLO post‑processing — DFL‑based bbox decoding + batch NMS.

Given head outputs (cls_scores, reg_preds) at multiple feature levels,
produces per‑image lists of [x1, y1, x2, y2, confidence, class_id] results.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.ops


def _dist_decode(
    reg_pred: torch.Tensor,    # [B, 4*reg_max, H, W]
    reg_max: int = 16,
) -> torch.Tensor:
    """Convert per‑level regression logits to l,t,r,b in pixels (stride‑relative)."""
    B, C, H, W = reg_pred.shape
    proj = torch.arange(reg_max, dtype=reg_pred.dtype, device=reg_pred.device)
    dist = reg_pred.view(B, 4, reg_max, H, W).softmax(2)          # [B, 4, reg_max, H, W]
    dist = (dist.permute(0, 3, 4, 1, 2).contiguous() @ proj)      # [B, H, W, 4]
    return dist.permute(0, 3, 1, 2).contiguous()                   # [B, 4, H, W]


def decode_predictions(
    cls_scores: list[torch.Tensor],     # each [B, nc, H_l, W_l]
    reg_preds: list[torch.Tensor],     # each [B, 4*reg_max, H_l, W_l]
    strides: list[int],
    reg_max: int = 16,
    conf_thr: float = 0.25,
    nms_thr: float = 0.65,
    max_det: int = 300,
) -> list[torch.Tensor]:
    """Decode multi‑level head outputs into per‑image detections.

    Returns
    -------
    list of [N_i, 6] tensors  (x1, y1, x2, y2, conf, cls) for each image.
    """
    B = cls_scores[0].shape[0]
    device = cls_scores[0].device
    dtype = cls_scores[0].dtype

    results: list[torch.Tensor] = []

    for img_idx in range(B):
        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for lvl, (cls_map, reg_map, s) in enumerate(zip(cls_scores, reg_preds, strides)):
            _, nc, h, w = cls_map.shape
            na = h * w

            # Grid
            gy, gx = torch.meshgrid(
                torch.arange(h, dtype=dtype, device=device),
                torch.arange(w, dtype=dtype, device=device),
                indexing="ij",
            )
            cx = (gx.reshape(-1) + 0.5) * s
            cy = (gy.reshape(-1) + 0.5) * s

            # Scores
            cls_single = cls_map[img_idx]                          # [nc, H, W]
            cls_single = cls_single.reshape(nc, na).permute(1, 0)  # [na, nc]
            cls_single = cls_single.sigmoid()

            # Keep anchors above threshold
            score, label = cls_single.max(dim=-1)                  # [na]
            keep = score > conf_thr
            if not keep.any():
                continue

            # Decode boxes
            reg_single = reg_map[img_idx]                          # [4*reg_max, H, W]
            lt = _dist_decode(reg_single.unsqueeze(0), reg_max).squeeze(0)  # [4, H, W]
            lt = lt.reshape(4, na).permute(1, 0)                   # [na, 4]  (l,t,r,b)

            # To xyxy in image coords
            cx_k = cx[keep]
            cy_k = cy[keep]
            lt_k = lt[keep] * s

            x1 = (cx_k - lt_k[:, 0]).clamp(min=0)
            y1 = (cy_k - lt_k[:, 1]).clamp(min=0)
            x2 = (cx_k + lt_k[:, 2]).clamp(min=0)
            y2 = (cy_k + lt_k[:, 3]).clamp(min=0)

            all_boxes.append(torch.stack([x1, y1, x2, y2], dim=-1))
            all_scores.append(score[keep])
            all_labels.append(label[keep])

        if not all_boxes:
            results.append(torch.zeros(0, 6, device=device, dtype=dtype))
            continue

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # NMS
        dets = batch_nms(boxes, scores, labels, nms_thr, max_det)
        results.append(dets)

    return results


def batch_nms(
    boxes: torch.Tensor,         # [N, 4] xyxy
    scores: torch.Tensor,        # [N]
    labels: torch.Tensor,        # [N] long
    iou_thr: float = 0.65,
    max_det: int = 300,
) -> torch.Tensor:
    """Per‑class NMS — returns [M, 6] (x1,y1,x2,y2,conf,cls)."""
    if boxes.numel() == 0:
        return torch.zeros(0, 6, device=boxes.device, dtype=boxes.dtype)

    unique_labels = labels.unique()
    out: list[torch.Tensor] = []

    for cls_id in unique_labels:
        m = labels == cls_id
        b = boxes[m]
        s = scores[m]

        keep = torchvision.ops.nms(b, s, iou_thr)
        n = min(len(keep), max_det)

        kept_boxes = b[keep[:n]]
        kept_scores = s[keep[:n]]
        cls_col = torch.full((n, 1), cls_id.float(), device=b.device, dtype=b.dtype)

        out.append(torch.cat([kept_boxes, kept_scores.unsqueeze(-1), cls_col], dim=-1))

    if not out:
        return torch.zeros(0, 6, device=boxes.device, dtype=boxes.dtype)

    dets = torch.cat(out, dim=0)
    # Sort by confidence
    dets = dets[dets[:, 4].argsort(descending=True)]
    return dets[:max_det]