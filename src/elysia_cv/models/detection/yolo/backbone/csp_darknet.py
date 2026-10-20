"""CSPDarkNet backbone with shared blocks for YOLOv8/v10/v11/v12/v26.

All YOLO variants share the same CSPDarkNet skeleton; the difference is which
block type populates each stage:

    v8 / v10  →  C2f  (Bottleneck-based CSP)
    v11 / v26 →  C3k2 (larger‑kernel CSP)
    v12       →  A2C2f (attention‑centric CSP, defined in a2c2f.py)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Primitive blocks
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Conv2d → BatchNorm2d → SiLU (the YOLO-standard building block)."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard residual bottleneck:  Conv(k3)→Conv(k3) + shortcut."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Module):
    """CSP Bottleneck with an extra 1×1 conv squeeze — used inside C3k2."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


# ---------------------------------------------------------------------------
# Stage blocks (CSP variants)
# ---------------------------------------------------------------------------

class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions + n Bottlenecks.  Used by v8 / v10."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)                     # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class C3k2(C2f):
    """Larger‑kernel CSP block used by v11 / v26.  Same structure as C2f but
    with C3k sub‑blocks instead of plain Bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, e)       # sets self.c, self.cv1, self.cv2
        c_ = self.c // 2 if n > 0 else self.c
        self.m = nn.ModuleList([C3k(self.c, self.c, 1, shortcut) for _ in range(n)])


# ---------------------------------------------------------------------------
# SPPF — Spatial Pyramid Pooling Fast
# ---------------------------------------------------------------------------

class SPPF(nn.Module):
    """Three cascaded 5×5 max‑pools (equivalent to 5/9/13 receptive fields)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ---------------------------------------------------------------------------
# CSPDarkNet assembly
# ---------------------------------------------------------------------------

class CSPDarkNet(nn.Module):
    """Shared backbone for YOLOv8/v10/v11/v12/v26.

    Parameters
    ----------
    block_type : {"c2f", "c3k2", "a2c2f"}
        Which CSP block to populate stages with.  "a2c2f" requires the A2C2f
        class to be passed separately (or injected after construction).
    depth_multiple : float, default 1.0
        Multiplier for the number of blocks per stage.
    width_multiple : float, default 1.0
        Multiplier for channel widths (rounded to multiple of 8).
    max_channels : int, default 1024
        Cap on channel width after width_multiple is applied.
    block_cls : type, optional
        Override the block class (used for "a2c2f").
    """

    # Base architecture: (out_channels, num_blocks) per stage + final SPPF
    _BASE = [
        (64, 3),       # stage 1 → P2  (1/4)
        (128, 6),      # stage 2 → P3  (1/8)
        (256, 6),      # stage 3 → P4  (1/16)
        (512, 3),      # stage 4 → P5  (1/32)
    ]

    def __init__(
        self,
        block_type: str = "c2f",
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0,
        max_channels: int = 1024,
        block_cls: type | None = None,
        out_indices: tuple[int, ...] = (2, 3, 4),   # return P3 / P4 / P5
    ):
        super().__init__()
        self.out_indices = out_indices

        # --- channel helpers ---
        def _make_divisible(x: float, divisor: int = 8) -> int:
            return max(round(x / divisor) * divisor, divisor)

        def _ch(c: int) -> int:
            return min(_make_divisible(c * width_multiple), max_channels)

        # --- pick block factory ---
        if block_cls is not None:
            _block = block_cls
        elif block_type == "c2f":
            _block = C2f
        elif block_type == "c3k2":
            _block = C3k2
        else:
            raise ValueError(f"Unknown block_type={block_type}; use block_cls= for a2c2f")

        # --- stem ---
        self.stem = Conv(3, _ch(64), 3, 2)          # /2

        # --- stage 1 ---
        self.stage1 = nn.Sequential(
            Conv(_ch(64), _ch(128), 3, 2),           # /4
            _block(_ch(128), _ch(128), max(1, int(3 * depth_multiple))),
        )

        # --- stage 2 → P3 ---
        self.stage2 = nn.Sequential(
            Conv(_ch(128), _ch(256), 3, 2),          # /8
            _block(_ch(256), _ch(256), max(1, int(6 * depth_multiple))),
        )

        # --- stage 3 → P4 ---
        self.stage3 = nn.Sequential(
            Conv(_ch(256), _ch(512), 3, 2),          # /16
            _block(_ch(512), _ch(512), max(1, int(6 * depth_multiple))),
        )

        # --- stage 4 → P5 ---
        self.stage4 = nn.Sequential(
            Conv(_ch(512), _ch(max_channels), 3, 2),  # /32
            _block(_ch(max_channels), _ch(max_channels), max(1, int(3 * depth_multiple))),
        )

        # --- SPPF (on P5) ---
        self.sppf = SPPF(_ch(max_channels), _ch(max_channels))

        self._out_channels = {
            idx: _ch(ch) for idx, (ch, _) in enumerate(self._BASE, start=1)
        }
        # Add SPPF output at stage 5 (same width as stage 4)
        self._out_channels[5] = _ch(max_channels)

    @property
    def out_channels(self) -> dict[int, int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Returns a tuple of multi‑scale features (P3, P4, P5 by default)."""
        feats: dict[int, torch.Tensor] = {}

        x = self.stem(x)                    # 1/2
        x = self.stage1(x)                  # 1/4  → P2 (not returned by default)
        feats[2] = x

        x = self.stage2(x)                  # 1/8  → P3
        feats[3] = x

        x = self.stage3(x)                  # 1/16 → P4
        feats[4] = x

        x = self.stage4(x)                  # 1/32 → P5
        x = self.sppf(x)
        feats[5] = x

        return tuple(feats[i] for i in self.out_indices)