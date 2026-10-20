"""YOLO variant definitions — depth, width, block types for every (n/s/m/l/x).

Each entry maps a variant key to:
    version   : "v8" | "v10" | "v11" | "v12" | "v26"
    block     : "c2f" | "c3k2" | "a2c2f"
    head      : "standard" | "v10_dual"
    depth     : float  → depth_multiple for CSPDarkNet / PAN / head blocks
    width     : float  → width_multiple for channels
    max_ch    : int    → max_channels cap

All models use 3 detection heads at strides 8 / 16 / 32.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class YOLOVariant:
    version: str            # "v8" | "v10" | "v11" | "v12" | "v26"
    block: str              # "c2f" | "c3k2" | "a2c2f"
    head: str               # "standard" | "v10_dual"
    depth: float            # depth_multiple
    width: float            # width_multiple
    max_ch: int             # max_channels
    desc: str = ""          # human‑readable description

    @property
    def key(self) -> str:
        return f"yolo{self.version}{self.depth_str}{self.width_str}"

    @property
    def depth_str(self) -> str:
        """e.g. 0.33→'n', 0.50→'s', 0.67→'m', 1.00→'l' or 'x'."""
        if self.depth <= 0.34:
            return "n"
        if self.depth <= 0.51:
            return "s" if self.width < 0.70 else "m"
        if self.depth <= 0.68:
            return "m"
        if self.width >= 1.2:
            return "x"
        return "l"

    @property
    def width_str(self) -> str:
        return self.depth_str  # simplified; depth+width always paired


# ── YOLOv10 (NMS‑free) ──
# v10 has no l/x; largest variant is m
_V10 = lambda d, w: YOLOVariant("v10", "c2f", "v10_dual", d, w, 1024)

# ── YOLOv11 (C3k2) ──
_V11 = lambda d, w: YOLOVariant("v11", "c3k2", "standard", d, w, 1024)

# ── YOLOv12 (A2C2f attention) ──
_V12 = lambda d, w: YOLOVariant("v12", "a2c2f", "standard", d, w, 1024)

# ── YOLO26 (C3k2, latest) ──
_V26 = lambda d, w: YOLOVariant("v26", "c3k2", "standard", d, w, 1024)


_VARIANTS: dict[str, YOLOVariant] = {
    # YOLOv10
    "yolov10n": _V10(0.33, 0.25),
    "yolov10s": _V10(0.33, 0.50),
    "yolov10m": _V10(0.67, 0.75),
    # YOLOv11
    "yolov11n": _V11(0.50, 0.25),
    "yolov11s": _V11(0.50, 0.50),
    "yolov11m": _V11(0.50, 0.75),
    "yolov11l": _V11(1.00, 1.00),
    "yolov11x": _V11(1.00, 1.25),
    # YOLOv12
    "yolov12n": _V12(0.50, 0.25),
    "yolov12s": _V12(0.50, 0.50),
    "yolov12m": _V12(0.50, 0.75),
    "yolov12l": _V12(1.00, 1.00),
    "yolov12x": _V12(1.00, 1.25),
    # YOLO26
    "yolov26n": _V26(0.33, 0.25),
    "yolov26s": _V26(0.33, 0.50),
    "yolov26m": _V26(0.67, 0.75),
    "yolov26l": _V26(1.00, 1.00),
    "yolov26x": _V26(1.00, 1.25),
}


def get_variant(name: str) -> YOLOVariant:
    """Look up a variant by its key (e.g. ``'yolov11n'``)."""
    if name not in _VARIANTS:
        raise KeyError(
            f"Unknown YOLO variant '{name}'. Available: {sorted(_VARIANTS.keys())}"
        )
    return _VARIANTS[name]


def list_variants() -> list[str]:
    """Return all registered variant keys."""
    return sorted(_VARIANTS.keys())