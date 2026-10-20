"""NMS‑free dual detection head for YOLOv10.

YOLOv10 trains with two parallel heads:
    one2many  — standard label assignment (used for training only)
    one2one   — Hungarian‑style one‑to‑one assignment (consistent‑match distillation)

At inference time only the one2one head is used, eliminating NMS entirely.

Both heads share the same architecture (decoupled cls + reg branches) but
receive different label assignments during training.

Reference
---------
YOLOv10: Real‑Time End‑to‑End Object Detection, 2024.
"""

import torch
import torch.nn as nn

from ..backbone.csp_darknet import Conv


def _make_branch(c1: int, c2: int, num: int) -> nn.Sequential:
    """Per‑level branch: Conv(k3) → Conv(k3) → Conv(k1)."""
    return nn.Sequential(
        Conv(c1, c1, 3, 1),
        Conv(c1, c1, 3, 1),
        nn.Conv2d(c1, num, 1),
    )


class V10DetectHead(nn.Module):
    """YOLOv10 NMS‑free detection head (one2many + one2one)."""

    def __init__(
        self,
        in_channels: tuple[int, ...] = (256, 256, 256),
        num_classes: int = 80,
        reg_max: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        nc = num_classes
        rmax = reg_max * 4

        # ── one2many (standard) ──
        self.o2m_cls = nn.ModuleList()
        self.o2m_reg = nn.ModuleList()

        # ── one2one (NMS‑free) ──
        self.o2o_cls = nn.ModuleList()
        self.o2o_reg = nn.ModuleList()

        for ch in in_channels:
            self.o2m_cls.append(_make_branch(ch, ch, nc))
            self.o2m_reg.append(_make_branch(ch, ch, rmax))
            self.o2o_cls.append(_make_branch(ch, ch, nc))
            self.o2o_reg.append(_make_branch(ch, ch, rmax))

    def forward(
        self,
        feats: tuple[torch.Tensor, ...],
        mode: str = "both",
    ) -> dict[str, list[torch.Tensor]]:
        """Returns dict with 'cls_o2m','reg_o2m' and/or 'cls_o2o','reg_o2o'.

        Parameters
        ----------
        mode : {"both", "o2m", "o2o"}
            - ``"both"`` — training; returns all four lists.
            - ``"o2o"``  — inference; returns one2one only.
        """
        result: dict[str, list[torch.Tensor]] = {}

        if mode in ("both", "o2m"):
            cls_o2m, reg_o2m = [], []
            for f, cb, rb in zip(feats, self.o2m_cls, self.o2m_reg):
                cls_o2m.append(cb(f))
                reg_o2m.append(rb(f))
            result["cls_o2m"], result["reg_o2m"] = cls_o2m, reg_o2m

        if mode in ("both", "o2o"):
            cls_o2o, reg_o2o = [], []
            for f, cb, rb in zip(feats, self.o2o_cls, self.o2o_reg):
                cls_o2o.append(cb(f))
                reg_o2o.append(rb(f))
            result["cls_o2o"], result["reg_o2o"] = cls_o2o, reg_o2o

        return result