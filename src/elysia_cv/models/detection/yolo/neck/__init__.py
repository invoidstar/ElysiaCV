"""PAN-FPN neck shared across YOLOv8/v10/v11/v12/v26.

Top‑down FPN + bottom‑up PAN with CSP blocks at each fusion point.
The block type (C2f / C3k2 / A2C2f) is injected via ``block_cls``.
"""

import torch
import torch.nn as nn

from ..backbone.csp_darknet import Conv, C2f


class PAN(nn.Module):
    """Path Aggregation Network (PAN‑FPN).

    Parameters
    ----------
    in_channels : tuple[int, ...]
        Input feature channels, typically (P3_ch, P4_ch, P5_ch).
    out_channels : int
        Output channels per level (all levels share the same width).
    block_cls : type, optional
        CSP block class (default ``C2f``).  Use ``C3k2`` for YOLOv11/v26,
        ``A2C2f`` for YOLOv12.
    depth_multiple : float
        Multiplier for the number of CSP blocks at each fusion.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = (256, 512, 1024),
        out_channels: int = 256,
        block_cls: type = C2f,
        depth_multiple: float = 1.0,
    ):
        super().__init__()

        nc = out_channels

        # ── Top‑down (FPN) ──
        self.lateral_p5 = Conv(in_channels[2], nc, 1, 1)
        self.lateral_p4 = Conv(in_channels[1], nc, 1, 1)
        self.lateral_p3 = Conv(in_channels[0], nc, 1, 1)

        self.fpn_p4 = block_cls(nc * 2, nc, max(1, int(3 * depth_multiple)))
        self.fpn_p3 = block_cls(nc * 2, nc, max(1, int(3 * depth_multiple)))

        # ── Bottom‑up (PAN) ──
        self.down_p3 = Conv(nc, nc, 3, 2)
        self.down_p4 = Conv(nc, nc, 3, 2)

        self.pan_p4 = block_cls(nc * 2, nc, max(1, int(3 * depth_multiple)))
        self.pan_p5 = block_cls(nc * 2, nc, max(1, int(3 * depth_multiple)))

    def forward(
        self, feats: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        p3, p4, p5 = feats

        # top‑down
        p5_fpn = self.lateral_p5(p5)                     # [B, nc, H/32, W/32]
        p4_fpn = self.lateral_p4(p4)                     # [B, nc, H/16, W/16]
        p3_fpn = self.lateral_p3(p3)                     # [B, nc, H/8,  W/8]

        p4_fpn = self.fpn_p4(torch.cat([
            p4_fpn,
            nn.functional.interpolate(p5_fpn, scale_factor=2.0, mode="nearest"),
        ], dim=1))

        p3_out = self.fpn_p3(torch.cat([
            p3_fpn,
            nn.functional.interpolate(p4_fpn, scale_factor=2.0, mode="nearest"),
        ], dim=1))

        # bottom‑up
        p4_pan = self.pan_p4(torch.cat([p4_fpn, self.down_p3(p3_out)], dim=1))
        p5_out = self.pan_p5(torch.cat([p5_fpn, self.down_p4(p4_pan)], dim=1))

        return p3_out, p4_pan, p5_out