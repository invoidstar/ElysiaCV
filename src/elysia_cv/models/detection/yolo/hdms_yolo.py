"""HDMS‑YOLO — High‑Density Multi‑Scale YOLO for agricultural weed detection.

Key improvements over standard YOLOv11:
1. **SimSPPF** — Uses SiLU activation only, replaces Conv+BN+SiLU with 
   Conv+SiLU in SPPF to reduce latency.

2. **EMA (Efficient Multi‑Scale Attention)** — Lightweight cross‑spatial
   attention that re‑weights features across channel and spatial dimensions.
   Improves detection of small / occluded weeds in dense canopies.

3. **Built on YOLOv11n baseline** — Reuses CSPDarkNet+C3k2+PAN+DetectHead
   with minimal modifications for agricultural deployment.

Reference
---------
HDMS‑YOLO: High‑Density Multi‑Scale Object Detection in Complex Farmland
Environments, 2025.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import YOLODetector, _block_cls_for
from .variants import get_variant, list_variants
from .backbone import CSPDarkNet
from .neck import PAN
from .head.detect_head import DetectHead


# ══════════════════════════════════════════════════════════════════════════════
# SimSPPF — SiLU‑only SPPF
# ══════════════════════════════════════════════════════════════════════════════

class SimConv(nn.Module):
    """Conv2d + SiLU (no BatchNorm) — reduced latency alternative to Conv."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SimSPPF(nn.Module):
    """SimSPPF — SiLU‑only SPPF (no BatchNorm, 5×5 max‑pool).

    Replaces standard Conv → Conv+SiLU for faster inference with negligible
    accuracy drop on agricultural datasets.
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = SimConv(c1, c_, 1)
        self.cv2 = SimConv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# EMA — Efficient Multi‑Scale Attention
# ══════════════════════════════════════════════════════════════════════════════

class EMA(nn.Module):
    """Efficient Multi‑Scale Attention — lightweight cross‑spatial attention.

    Applies parallel 1×1 + 3×3 depthwise branches, channels are split into
    sub‑groups (each branch processes a subset), cross‑spatial pooling
    re‑weights, and a channel‑attention gate provides the final scale.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    factor : int
        Channel reduction factor for the bottleneck, default 8.
    """

    def __init__(self, channels: int, factor: int = 8, groups: int = 1):
        super().__init__()
        self.groups = max(groups, 1)

        # Parallel branches — each processes all channels
        self.conv_1x1 = nn.Conv2d(channels, channels, 1)
        self.conv_3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv_fuse = nn.Conv2d(channels, channels, 1)

        # Channel attention bottleneck
        hidden = max(channels // factor, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

        # Cross‑spatial pooling (height & width)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Channel attention gate ──
        att = self.gap(x)
        att = self.fc1(att)
        att = F.silu(att)
        att = self.fc2(att).sigmoid()       # [B, C, 1, 1]

        # ── Parallel branches ──
        # Split channels in half for two branches
        c_half = C // 2
        x1 = x[:, :c_half]                   # [B, c_half, H, W]

        b1 = self.conv_1x1(x)                # 1×1 on full channels
        b1_half = b1[:, :c_half]             # take half for spatial attention

        # Cross‑spatial re‑weight
        h_pool = self.pool_h(b1_half).sigmoid()   # [B, c_half, H, 1]
        w_pool = self.pool_w(b1_half).sigmoid()   # [B, c_half, 1, W]
        b1_spatial = b1_half * h_pool * w_pool

        b2 = self.conv_3x3(x)               # 3×3 depthwise on full channels
        b2_other = b2[:, c_half:]            # take other half

        # Combine: spatial half from 1×1 + second half from 3×3
        out = torch.cat([b1_spatial, b2_other], dim=1)  # [B, C, H, W]
        out = self.conv_fuse(out)

        return out * att


# ══════════════════════════════════════════════════════════════════════════════
# HDMS‑YOLO model builder
# ══════════════════════════════════════════════════════════════════════════════

class HDMSDetector(YOLODetector):
    """HDMS‑YOLO detector — YOLOv11 backbone + SimSPPF + EMA neck + standard head.

    Builds on YOLOv11n/s/m variants with:
    - SimSPPF replacing standard SPPF (reduced latency)
    - EMA attention inserted at P3/P4/P5 feature levels
    - Same DetectHead and loss as YOLOv11
    """

    def __init__(self, variant: str = "yolov11n", num_classes: int = 80):
        v = get_variant(variant)
        super().__init__(variant=variant, num_classes=num_classes)

        # Replace SPPF with SimSPPF
        bb_ch = self.backbone.out_channels
        p5_ch = bb_ch[5]
        self.backbone.sppf = SimSPPF(p5_ch, p5_ch)

        # Insert EMA after P3/P4/P5 (before neck)
        p3_ch = bb_ch[3]
        p4_ch = bb_ch[4]
        self.ema_p3 = EMA(p3_ch)
        self.ema_p4 = EMA(p4_ch)
        self.ema_p5 = EMA(p5_ch)

    def forward(self, x: torch.Tensor, mode: str = "both") -> dict[str, list[torch.Tensor]]:
        """Backbone → EMA attention → neck → head."""
        feats = self.backbone(x)          # (P3, P4, P5)

        # Apply EMA attention
        p3 = self.ema_p3(feats[0])
        p4 = self.ema_p4(feats[1])
        p5 = self.ema_p5(feats[2])

        feats = self.neck((p3, p4, p5))
        cls_scores, reg_preds = self.head(feats)
        return {"cls_scores": cls_scores, "reg_preds": reg_preds}


def build_hdms(variant: str = "yolov11n", num_classes: int = 80) -> HDMSDetector:
    """Convenience factory for HDMS‑YOLO."""
    return HDMSDetector(variant, num_classes=num_classes)