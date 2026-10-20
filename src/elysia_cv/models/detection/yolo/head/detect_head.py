"""Decoupled detection head for YOLOv8/v11/v12/v26.

Each feature level (P3/P4/P5) gets an independent pair of branches:
    cls_branch : Conv(k3)→Conv(k3)→Conv(k1) → class logits
    reg_branch : Conv(k3)→Conv(k3)→Conv(k1) → bbox distribution (reg_max bins)

The head uses Distribution Focal Loss (DFL) where each of the 4 bbox
coordinates is modelled by ``reg_max`` soft‑classified bins.
"""

import torch
import torch.nn as nn

from ..backbone.csp_darknet import Conv


class DetectHead(nn.Module):
    """Standard YOLO decoupled detection head.

    Parameters
    ----------
    in_channels : tuple[int, ...]
        Channels per feature level (P3, P4, P5), normally [nc, nc, nc].
    num_classes : int
        Number of classes (excluding background).
    reg_max : int, default 16
        Number of bins for DFL regression per coordinate.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = (256, 256, 256),
        num_classes: int = 80,
        reg_max: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.no = num_classes + reg_max * 4          # output dim per anchor‑point

        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()

        for ch in in_channels:
            self.cls_branches.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1),
                nn.Conv2d(ch, num_classes, 1),
            ))
            self.reg_branches.append(nn.Sequential(
                Conv(ch, ch, 3, 1),
                Conv(ch, ch, 3, 1),
                nn.Conv2d(ch, reg_max * 4, 1),
            ))

    def forward(
        self,
        feats: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cls_scores, reg_preds) as lists of per‑level tensors."""
        cls_scores: list[torch.Tensor] = []
        reg_preds: list[torch.Tensor] = []

        for f, cls_br, reg_br in zip(feats, self.cls_branches, self.reg_branches):
            cls_scores.append(cls_br(f))
            reg_preds.append(reg_br(f))

        return cls_scores, reg_preds