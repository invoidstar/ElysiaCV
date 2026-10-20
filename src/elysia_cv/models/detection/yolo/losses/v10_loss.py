"""YOLOv10 NMS‑free training loss — one2many + one2one dual loss.

YOLOv10 uses two parallel heads:
    one2many (O2M)  — standard TAL assignment, same as YOLOv8
    one2one  (O2O)  — per‑GT one‑to‑one matching (Hungarian‑style)

At inference only O2O is used → no NMS needed.

The total loss is:
    L_total = w_o2m * L(o2m) + w_o2o * L(o2o)

where L(·) = w_cls * BCE + w_box * CIoU + w_dfl * DFL
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v8_loss import _bbox_iou, _ciou_loss, _dfl_loss  # shared helper fns
from .task_aligned_assigner import TaskAlignedAssigner


class V10Loss(nn.Module):
    """YOLOv10 dual‑head loss.

    Parameters
    ----------
    num_classes : int
    reg_max : int, default 16
    strides : list[int]
    tal_topk : int, default 13
    loss_weights : dict
        Keys: ``"cls"``, ``"box"``, ``"dfl"``, ``"o2m"``, ``"o2o"``.
        Default: cls=0.5, box=7.5, dfl=1.5, o2m=1.0, o2o=1.0
    """

    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        strides: list[int] | None = None,
        tal_topk: int = 13,
        loss_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.strides = strides or [8, 16, 32]

        # Two assigners: O2M uses standard TAL, O2O uses top-1
        self.assigner_o2m = TaskAlignedAssigner(topk=tal_topk, nc=num_classes)
        self.assigner_o2o = TaskAlignedAssigner(topk=1, nc=num_classes)  # one‑to‑one

        self.loss_weights = loss_weights or {
            "cls": 0.5, "box": 7.5, "dfl": 1.5, "o2m": 1.0, "o2o": 0.5,
        }
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        cls_o2m: list[torch.Tensor],     # [B,nc,H,W] × 3
        reg_o2m: list[torch.Tensor],
        cls_o2o: list[torch.Tensor],
        reg_o2o: list[torch.Tensor],
        gt_bboxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute dual loss.

        Returns ``{"loss": total, "losses": {...}, "metrics": {...}}``.
        """
        loss_o2m, losses_o2m, metrics_o2m = self._single_loss(
            cls_o2m, reg_o2m, gt_bboxes, gt_labels, self.assigner_o2m
        )
        loss_o2o, losses_o2o, metrics_o2o = self._single_loss(
            cls_o2o, reg_o2o, gt_bboxes, gt_labels, self.assigner_o2o
        )

        w_o2m = self.loss_weights.get("o2m", 1.0)
        w_o2o = self.loss_weights.get("o2o", 0.5)
        total = w_o2m * loss_o2m + w_o2o * loss_o2o

        return {
            "loss": total,
            "losses": {
                "cls_o2m": losses_o2m["cls_loss"],
                "box_o2m": losses_o2m["box_loss"],
                "dfl_o2m": losses_o2m["dfl_loss"],
                "cls_o2o": losses_o2o["cls_loss"],
                "box_o2o": losses_o2o["box_loss"],
                "dfl_o2o": losses_o2o["dfl_loss"],
            },
            "metrics": {
                "o2m_iou": metrics_o2m["iou"],
                "o2o_iou": metrics_o2o["iou"],
                "o2m_fg": metrics_o2m["fg_ratio"],
                "o2o_fg": metrics_o2o["fg_ratio"],
            },
        }

    # ── internal single‑path loss (shared with v8_loss logic) ─────────────

    def _single_loss(
        self,
        cls_scores: list[torch.Tensor],
        reg_preds: list[torch.Tensor],
        gt_bboxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        assigner: TaskAlignedAssigner,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Compute loss for ONE head path (O2M or O2O)."""
        B = len(cls_scores[0])
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype

        # Flatten
        anchors, cls_all, reg_all, stride_all = [], [], [], []
        for lvl, (c, r, s) in enumerate(zip(cls_scores, reg_preds, self.strides)):
            _, _, h, w = c.shape
            na_l = h * w
            gy, gx = torch.meshgrid(
                torch.arange(h, dtype=dtype, device=device),
                torch.arange(w, dtype=dtype, device=device), indexing="ij",
            )
            ap = torch.stack([(gx+0.5)*s, (gy+0.5)*s], dim=-1).reshape(1, na_l, 2).expand(B, -1, -1)
            anchors.append(ap)
            cls_all.append(c.reshape(B, self.nc, na_l).permute(0, 2, 1))
            reg_all.append(r.reshape(B, self.reg_max*4, na_l).permute(0, 2, 1))
            stride_all.append(torch.full((1, na_l, 1), s, dtype=dtype, device=device).expand(B, -1, -1))

        ap_all = torch.cat(anchors, 1)         # [B, total_na, 2]
        cls_flat = torch.cat(cls_all, 1)         # [B, total_na, nc]
        reg_flat = torch.cat(reg_all, 1)         # [B, total_na, 4*reg_max]
        st_flat = torch.cat(stride_all, 1)       # [B, total_na, 1]

        # Pad GTs
        max_gt = max(len(g) for g in gt_labels)
        gt_lab_t = torch.full((B, max_gt, 1), self.nc, dtype=torch.long, device=device)
        gt_box_t = torch.zeros(B, max_gt, 4, dtype=dtype, device=device)
        mask_gt = torch.zeros(B, max_gt, dtype=torch.bool, device=device)
        for b in range(B):
            n = len(gt_labels[b])
            if n == 0: continue
            gt_lab_t[b, :n, 0] = gt_labels[b][:n].long()
            gt_box_t[b, :n, :] = gt_bboxes[b][:n]
            mask_gt[b, :n] = True

        # Decode predicted boxes (for TAL)
        pred_dist_4 = reg_flat.view(B, -1, 4, self.reg_max)
        proj = torch.arange(self.reg_max, dtype=dtype, device=device)
        pred_lt = (pred_dist_4.softmax(-1) @ proj) * st_flat
        pred_boxes = torch.cat([
            ap_all[..., :2] - pred_lt[..., :2],
            ap_all[..., :2] + pred_lt[..., 2:],
        ], dim=-1)

        # Assign
        tgt_lab, tgt_box, tgt_score, fg_mask = assigner(
            cls_flat, pred_boxes, ap_all, gt_lab_t, gt_box_t, mask_gt,
        )

        # ── cls loss ──
        tgt_cls = F.one_hot(tgt_lab.clamp(0, self.nc-1), self.nc).float()
        w_cls = (fg_mask.unsqueeze(-1).float() * tgt_score.unsqueeze(-1)).detach()
        cls_loss = self.bce(cls_flat, tgt_cls)
        cls_loss = (cls_loss * w_cls).sum() / max(w_cls.sum(), 1)

        # ── box loss ──
        if fg_mask.any():
            iou = _bbox_iou(pred_boxes[fg_mask], tgt_box[fg_mask]).detach()
            box_loss = _ciou_loss(pred_boxes[fg_mask], tgt_box[fg_mask]).mean()
        else:
            iou = torch.tensor(0.0, device=device)
            box_loss = torch.tensor(0.0, device=device)

        # ── DFL loss ──
        if fg_mask.any():
            pred_lt_fg = pred_lt[fg_mask]
            tgt_lt_fg = torch.cat([
                ap_all[..., :2] - tgt_box[..., :2],
                tgt_box[..., 2:] - ap_all[..., :2],
            ], dim=-1).clamp(min=0)[fg_mask]
            dfl = _dfl_loss(reg_flat[fg_mask], tgt_lt_fg, st_flat[fg_mask], self.reg_max)
        else:
            dfl = torch.tensor(0.0, device=device)

        # Sum
        wc = self.loss_weights.get("cls", 0.5)
        wb = self.loss_weights.get("box", 7.5)
        wd = self.loss_weights.get("dfl", 1.5)
        loss = wc * cls_loss + wb * box_loss + wd * dfl

        losses = {"cls_loss": cls_loss, "box_loss": box_loss, "dfl_loss": dfl}
        metrics = {"iou": iou.mean() if fg_mask.any() else iou,
                    "fg_ratio": fg_mask.float().mean()}
        return loss, losses, metrics