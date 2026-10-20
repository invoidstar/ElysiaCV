"""A2C2f — Attention‑centric CSP block for YOLOv12.

YOLOv12 replaces the standard Bottleneck in C2f with an area‑attention
mechanism that splits the feature map into N equal‑area regions, applies
self‑attention within each region, and fuses the output back through a
Conv‑BN‑SiLU layer.

Reference
---------
YOLOv12: Attention‑Centric Real‑Time Object Detectors, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .csp_darknet import Conv


class AreaAttention(nn.Module):
    """Divide feature map into N equal‑area patches, apply scaled dot‑product
    attention within each patch, and reassemble."""

    def __init__(self, dim: int, num_heads: int = 4, area: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.area = area
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv = Conv(dim, dim * 3, 1, 1)
        self.proj = Conv(dim, dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        area = self.area
        nh = self.num_heads
        hd = self.head_dim

        # Pad to multiple of area
        pad_h = (area - H % area) % area
        pad_w = (area - W % area) % area
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H_p, W_p = x.shape
        else:
            H_p, W_p = H, W

        # qkv projection
        qkv = self.qkv(x)                                   # [B, 3C, H_p, W_p]
        q, k, v = qkv.chunk(3, dim=1)                        # each [B, C, H_p, W_p]

        # Reshape into area×area patches
        gh, gw = H_p // area, W_p // area
        def _reshape(t: torch.Tensor) -> torch.Tensor:
            # [B, C, H_p, W_p] → [B, area*area, gh*gw, C]
            t = t.view(B, nh, hd, area, gh, area, gw)
            t = t.permute(0, 1, 3, 5, 4, 6, 2).contiguous()  # [B, nh, area, area, gh, gw, hd]
            t = t.view(B, nh, area * area, gh * gw, hd)       # [B, nh, N, L, hd]
            return t

        q = _reshape(q)
        k = _reshape(k)
        v = _reshape(v)

        # Scaled dot-product attention per head
        scale = hd ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v                                    # [B, nh, N, L, hd]

        # Reassemble spatial
        out = out.view(B, nh, area, area, gh, gw, hd)
        out = out.permute(0, 1, 6, 2, 4, 3, 5).contiguous()   # [B, nh, hd, area, gh, area, gw]
        out = out.view(B, C, area * gh, area * gw)             # [B, C, H_p, W_p]

        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return self.proj(out)


class A2C2fBlock(nn.Module):
    """Single A2C2f sub‑block: AreaAttention → Conv."""

    def __init__(self, c: int, shortcut: bool = True, num_heads: int = 4, area: int = 2):
        super().__init__()
        self.attn = AreaAttention(c, num_heads, area)
        self.cv = Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv(self.attn(x))
        return x + out if self.add else out


class A2C2f(nn.Module):
    """YOLOv12 CSP block — same topology as C2f but with A2C2fBlock sub‑blocks.

    Compatible with CSPDarkNet (pass block_cls=A2C2f) and PAN neck.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        num_heads: int = 4,
        area: int = 2,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1, 1)
        self.m = nn.ModuleList(
            [A2C2fBlock(c_, shortcut, num_heads, area) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))