"""YOLO model assembly — combines backbone, neck, and head for all variants.

Usage
-----
    from elysia_cv.models.detection.yolo import build_yolo

    model = build_yolo("yolov11n", num_classes=80)
    feats = model(torch.randn(1, 3, 640, 640))
    # → dict with cls_scores / reg_preds (list of per‑level tensors)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import CSPDarkNet, C2f, C3k2, A2C2f
from .neck import PAN
from .head.detect_head import DetectHead
from .head.v10_head import V10DetectHead
from .variants import get_variant, YOLOVariant


def _make_divisible(x: float, divisor: int = 8) -> int:
    return max(round(x / divisor) * divisor, divisor)


def _block_cls_for(variant: YOLOVariant) -> type:
    if variant.block == "c2f":
        return C2f
    elif variant.block == "c3k2":
        return C3k2
    elif variant.block == "a2c2f":
        return A2C2f
    raise ValueError(f"Unknown block type: {variant.block}")


class YOLODetector(nn.Module):
    """Assembled YOLO detector (backbone + neck + head).

    Parameters
    ----------
    variant : str | YOLOVariant
        Variant key (e.g. ``"yolov11n"``) or a pre‑built ``YOLOVariant``.
    num_classes : int
        Number of classes.
    """

    def __init__(self, variant: str | YOLOVariant, num_classes: int = 80):
        super().__init__()

        if isinstance(variant, str):
            variant = get_variant(variant)
        self.variant = variant
        self.num_classes = num_classes

        block = _block_cls_for(variant)
        wm = variant.width

        # --- backbone ---
        self.backbone = CSPDarkNet(
            block_type=variant.block,
            depth_multiple=variant.depth,
            width_multiple=wm,
            max_channels=variant.max_ch,
            block_cls=block,
            out_indices=(3, 4, 5),             # P3, P4, P5
        )

        # derive neck input channels from backbone's out_channels
        bb_ch = self.backbone.out_channels
        p3_ch = bb_ch[3]   # e.g. wm=0.25 → 128 / wm=1.0 → 256
        p4_ch = bb_ch[4]
        p5_ch = bb_ch[5]

        neck_in = (p3_ch, p4_ch, p5_ch)
        neck_out = _make_divisible(256 * wm)

        # --- neck ---
        self.neck = PAN(
            in_channels=neck_in,
            out_channels=neck_out,
            block_cls=block,
            depth_multiple=variant.depth,
        )

        head_in = (neck_out, neck_out, neck_out)

        # --- head ---
        if variant.head == "v10_dual":
            self.head = V10DetectHead(in_channels=head_in, num_classes=num_classes)
        else:
            self.head = DetectHead(in_channels=head_in, num_classes=num_classes)

    def forward(
        self, x: torch.Tensor, mode: str = "both"
    ) -> dict[str, list[torch.Tensor]]:
        """Run backbone→neck→head.

        Returns ``{"cls_scores": [...], "reg_preds": [...]}`` for standard
        heads, or ``{"cls_o2m":[...],"reg_o2m":[...],"cls_o2o":[...],"reg_o2o":[...]}``
        for YOLOv10 dual head when ``mode="both"``.
        """
        feats = self.backbone(x)                    # (P3, P4, P5)
        feats = self.neck(feats)

        if isinstance(self.head, V10DetectHead):
            return self.head(feats, mode=mode)

        cls_scores, reg_preds = self.head(feats)
        return {"cls_scores": cls_scores, "reg_preds": reg_preds}


def build_yolo(variant: str, num_classes: int = 80, pretrained: bool = False) -> YOLODetector:
    """Convenience factory — builds and optionally loads pretrained weights."""
    model = YOLODetector(variant, num_classes)
    if pretrained:
        # P2 will implement weight loading
        raise NotImplementedError("Pretrained weight loading — coming in P2")
    return model