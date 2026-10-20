"""YOLO runtime — build_model, forward_train, forward_test.

Gluing layer between the pure‑PyTorch YOLO model and the ElysiaCV
TrainerLoop contract. Handles both standard (v8/v11/v12/v26) and
NMS‑free (v10) heads transparently.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .model import build_yolo, YOLODetector
from .hdms_yolo import build_hdms, HDMSDetector
from .losses import YOLOv8Loss, V10Loss

logger = logging.getLogger(__name__)

_STRIDES = [8, 16, 32]


# ── Build ────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """Build a YOLO detector from ElysiaCV config."""
    mc = cfg.get("model", {})
    variant = mc.get("variant", "yolov11n")
    num_classes = mc.get("num_classes", 80)
    pretrained = mc.get("pretrained", False)
    pretrained_path = mc.get("pretrained_path", None)

    # HDMS variants use a separate builder
    if variant.startswith("hdms_"):
        base = variant[5:]  # hdms_yolov11n → yolov11n
        model = build_hdms(base, num_classes=num_classes)
    else:
        model = build_yolo(variant, num_classes=num_classes, pretrained=False)

    if pretrained and pretrained_path:
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        from .weights import load_weights
        load_weights(model, pretrained_path, variant_name=variant)

    lw = mc.get("loss_weights", {"cls": 0.5, "box": 7.5, "dfl": 1.5, "o2m": 1.0, "o2o": 0.5})
    model._loss_weights = lw  # type: ignore[attr-defined]

    if variant.startswith("yolov10"):
        model._yolo_loss = V10Loss(num_classes=num_classes, strides=_STRIDES, loss_weights=lw)  # type: ignore[attr-defined]
    else:
        model._yolo_loss = YOLOv8Loss(num_classes=num_classes, strides=_STRIDES, loss_weights=lw)  # type: ignore[attr-defined]

    return model


# ── Train ────────────────────────────────────────────────────────────────────

def forward_train(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Training forward — standard (v8/v11/v12/v26) and dual‑head (v10)."""
    if device is None:
        device = next(model.parameters()).device

    images = batch["inputs"].to(device)
    gt_bboxes = [b.to(device) for b in batch.get("gt_bboxes", [])]
    gt_labels = [l.to(device) for l in batch.get("gt_labels", [])]

    with torch.autocast(device_type=device.type, enabled=images.dtype == torch.float16):
        head_out = model(images)

    loss_module = getattr(model, "_yolo_loss")
    if loss_weights is not None:
        loss_module.loss_weights = loss_weights

    if isinstance(loss_module, V10Loss):
        # YOLOv10 dual head
        loss_dict = loss_module(
            head_out["cls_o2m"], head_out["reg_o2m"],
            head_out["cls_o2o"], head_out["reg_o2o"],
            gt_bboxes, gt_labels,
        )
    else:
        loss_dict = loss_module(
            head_out["cls_scores"], head_out["reg_preds"],
            gt_bboxes, gt_labels, images,
        )
    return loss_dict


# ── Test / Infer ─────────────────────────────────────────────────────────────

def forward_test(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Eval / inference — standard NMS for v8/v11/v12/v26; NMS‑free for v10."""
    if device is None:
        device = next(model.parameters()).device

    images = batch["inputs"].to(device)

    with torch.no_grad():
        head_out = model(images)

    if isinstance(model, YOLODetector) and model.variant.head == "v10_dual":
        # YOLOv10: use O2O head, no NMS
        dets = _v10_postprocess(head_out["cls_o2o"], head_out["reg_o2o"])
        return {"pred_instances": dets, "cls_scores": head_out["cls_o2o"], "reg_preds": head_out["reg_o2o"]}

    from .postprocess import decode_predictions
    dets = decode_predictions(
        head_out["cls_scores"], head_out["reg_preds"],
        strides=_STRIDES, conf_thr=0.25, nms_thr=0.65, max_det=300,
    )
    return {"pred_instances": dets, "cls_scores": head_out["cls_scores"], "reg_preds": head_out["reg_preds"]}


def _v10_postprocess(
    cls_list: list[torch.Tensor],
    reg_list: list[torch.Tensor],
    conf_thr: float = 0.25,
) -> list[torch.Tensor]:
    """YOLOv10 NMS‑free post‑processing — per‑image top‑300 detections."""
    B = cls_list[0].shape[0]
    device = cls_list[0].device
    dtype = cls_list[0].dtype
    reg_max = 16
    proj = torch.arange(reg_max, dtype=dtype, device=device)
    results: list[torch.Tensor] = []

    for b in range(B):
        boxes_l, scores_l, labels_l = [], [], []
        for lvl, (cls_map, reg_map, s) in enumerate(zip(cls_list, reg_list, _STRIDES)):
            _, nc, h, w = cls_map.shape
            cls_s = cls_map[b].sigmoid().reshape(nc, -1).t()  # [na, nc]

            # Decode
            reg_s = reg_map[b].view(4, reg_max, h, w).softmax(1).permute(1, 2, 3, 0).reshape(-1, 4, reg_max)
            lt = (reg_s @ proj).squeeze(-1) * s              # [na, 4]  l,t,r,b

            gy, gx = torch.meshgrid(
                torch.arange(h, dtype=dtype, device=device),
                torch.arange(w, dtype=dtype, device=device), indexing="ij",
            )
            cx = (gx.reshape(-1) + 0.5) * s
            cy = (gy.reshape(-1) + 0.5) * s

            x1 = (cx - lt[:, 0]).clamp(min=0)
            y1 = (cy - lt[:, 1]).clamp(min=0)
            x2 = (cx + lt[:, 2]).clamp(min=0)
            y2 = (cy + lt[:, 3]).clamp(min=0)

            score, label = cls_s.max(dim=-1)
            keep = score > conf_thr
            if not keep.any():
                continue
            boxes_l.append(torch.stack([x1[keep], y1[keep], x2[keep], y2[keep]], dim=-1))
            scores_l.append(score[keep])
            labels_l.append(label[keep])

        if not boxes_l:
            results.append(torch.zeros(0, 6, device=device, dtype=dtype))
            continue

        all_b = torch.cat(boxes_l, 0)
        all_s = torch.cat(scores_l, 0)
        all_l = torch.cat(labels_l, 0)
        dets = torch.cat([all_b, all_s.unsqueeze(-1), all_l.float().unsqueeze(-1)], dim=-1)
        dets = dets[dets[:, 4].argsort(descending=True)][:300]
        results.append(dets)

    return results


# ── Convenience ──────────────────────────────────────────────────────────────

def infer(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device | None = None,
    conf_thr: float = 0.25,
) -> torch.Tensor | list[torch.Tensor]:
    """Run detection on a single image."""
    if device is None:
        device = next(model.parameters()).device
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        head_out = model(image)

    if isinstance(model, YOLODetector) and model.variant.head == "v10_dual":
        return _v10_postprocess(head_out["cls_o2o"], head_out["reg_o2o"], conf_thr)

    from .postprocess import decode_predictions
    return decode_predictions(
        head_out["cls_scores"], head_out["reg_preds"],
        strides=_STRIDES, conf_thr=conf_thr, nms_thr=0.65, max_det=300,
    )