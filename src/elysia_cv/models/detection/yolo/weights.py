"""Load pretrained weights from ultralytics .pt checkpoints.

Two strategies:
1. **Direct key remapping** — handles stem, SPPF, head with regex rules
2. **Shape + signal‑path matching** — for backbone stages and PAN blocks

Also provides ``download_weights()`` to fetch official .pt files from GitHub.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.request
from typing import Any

import torch
import torch.nn as nn

from .model import YOLODetector
from .variants import get_variant, YOLOVariant

logger = logging.getLogger(__name__)

# ── Official weight URLs ─────────────────────────────────────────────────────

_WEIGHT_URLS: dict[str, str] = {
    # YOLOv11 (ultralytics releases)
    "yolov11n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolov11s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolov11m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "yolov11l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "yolov11x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    # YOLOv10
    "yolov10n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt",
    "yolov10s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10s.pt",
    "yolov10m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt",
    # YOLOv12
    "yolov12n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt",
    "yolov12s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt",
    "yolov12m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt",
    "yolov12l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt",
    "yolov12x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt",
    # YOLO26 — architecture compatible with YOLOv11, weights loadable from v11 .pt
    # Official pretrained weights not yet released; use yolov11n.pt as initialisation
    "yolov26n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolov26s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "yolov26m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
    "yolov26l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "yolov26x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
}

# ── Direct key remapping — stem / SPPF / head branches ───────────────────────

def _remap_direct(
    ult_k: str, variant: YOLOVariant,
) -> str | None:
    """Map well‑known ultralytics keys to ElysiaCV keys."""
    # Strip "model." prefix
    k = ult_k[6:] if ult_k.startswith("model.") else ult_k

    # Stem: model.0.conv / model.0.bn
    if re.match(r"0\.conv\.", k):
        return f"backbone.stem.conv.{k[8:]}"
    if re.match(r"0\.bn\.", k):
        return f"backbone.stem.bn.{k[6:]}"

    return None  # fall through to shape matching


# ── Shape‑based + signal‑path matching ───────────────────────────────────────

def _collect_our_params(model: nn.Module) -> dict[str, torch.Size]:
    return {n: p.shape for n, p in model.named_parameters()}


def _tail_match(ult_k: str, our_k: str, depth: int = 3) -> bool:
    """Check if the last `depth` dot‑separated parts match."""
    up = ult_k.split(".")
    op = our_k.split(".")
    for d in range(1, depth + 1):
        if d <= len(up) and d <= len(op) and up[-d:] == op[-d:]:
            return True
    return False


def _sequential_match(
    ult_params: list[tuple[str, torch.Tensor]],
    our_params: dict[str, torch.Size],
    depth_multiple: float,
) -> dict[str, str]:
    """Match ultralytics sequential params to ElysiaCV sequential params.

    ultralytics models use flat sequential indexing (model.0, model.1, ...).
    ElysiaCV uses hierarchical naming (backbone.stage1.0.cv1.conv.weight).

    Strategy:
    1. Build two flat lists sorted by position
    2. For each unique shape in ultralytics list, find the next unmatched
       ElysiaCV param of the same shape
    """
    # Our params in module order (sorted by name naturally)
    our_ordered = sorted(our_params.items())

    # Group our params by shape for fast lookup
    by_shape: dict[tuple, list[tuple[int, str]]] = {}
    for idx, (name, shape) in enumerate(our_ordered):
        by_shape.setdefault(tuple(shape), []).append((idx, name))

    # Track which our params have been used
    used_our: set[str] = set()
    remap: dict[str, str] = {}

    for ult_k, v in ult_params:
        vs = tuple(v.shape)
        if vs not in by_shape:
            continue
        # Find first unused candidate
        for _idx, o_name in by_shape[vs]:
            if o_name not in used_our:
                remap[ult_k] = o_name
                used_our.add(o_name)
                break

    return remap


# ── Ultralytics‑free .pt extraction ──────────────────────────────────────────

def _extract_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Extract raw state_dict from a .pt checkpoint.

    Handles ultralytics .pt files even when ``ultralytics`` is not installed,
    by temporarily injecting a stub ``ultralytics`` module into sys.modules.
    """
    # Pass 1: normal torch.load
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return _unpack_sd(obj)
    except ModuleNotFoundError:
        pass

    # Pass 2: inject fake ultralytics modules so pickle can resolve class refs
    import sys, types
    _needs_cleanup = "ultralytics" not in sys.modules

    class _Stub:
        def __init__(self, *a, **kw): pass
        def __setstate__(self, s): self._sd = s
        def state_dict(self, *a, **kw):
            return _walk_stub_sd(self._sd)

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return _Stub()

    class _StubModule(types.ModuleType):
        """Module that returns _Stub for any attribute access."""
        def __getattr__(self, name):
            return _Stub

    if _needs_cleanup:
        # All known ultralytics submodules used in .pt files
        modules = [
            "ultralytics",
            "ultralytics.utils",
            "ultralytics.utils.loss",
            "ultralytics.nn",
            "ultralytics.nn.modules",
            "ultralytics.nn.modules.block",
            "ultralytics.nn.modules.conv",
            "ultralytics.nn.modules.head",
            "ultralytics.nn.modules.transformer",
            "ultralytics.nn.modules.utils",
            "ultralytics.nn.tasks",
        ]
        for mod_name in modules:
            mod = types.ModuleType(mod_name)
            mod.__class__ = _StubModule  # overrides __getattr__
            mod.__path__ = []
            sys.modules[mod_name] = mod

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return _unpack_sd(obj)
    finally:
        if _needs_cleanup:
            for k in list(sys.modules):
                if k.startswith("ultralytics"):
                    del sys.modules[k]


def _walk_stub_sd(obj, prefix="") -> dict[str, torch.Tensor]:
    """Recursively collect all tensors from a _Stub-wrapped model tree.

    Handles three node types:
    1. Real ``nn.Module`` → collect params in-place, recurse children
    2. ``_Stub`` instance → delegate to ``_sd`` (the real module's __dict__)
    3. ``dict`` (the module's __dict__) → extract _parameters, recurse _modules
    """
    result: dict[str, torch.Tensor] = {}

    # ── Real nn.Module ──
    if isinstance(obj, torch.nn.Module):
        for name, param in obj.named_parameters(recurse=False):
            result[prefix + name] = param.data
        for name, child in obj.named_children():
            result.update(_walk_stub_sd(child, prefix + name + "."))
        return result

    # ── _Stub instance → use its _sd ──
    if hasattr(obj, "_sd") and not isinstance(obj, dict):
        return _walk_stub_sd(obj._sd, prefix)

    # ── dict (module __dict__) ──
    if isinstance(obj, dict):
        # Parameters: OrderedDict of name → Parameter/tensor
        params = obj.get("_parameters", {})
        for k, v in (params.items() if isinstance(params, dict) else []):
            if isinstance(v, torch.Tensor):
                result[prefix + k] = v
            elif hasattr(v, "data"):
                result[prefix + k] = v.data

        # Recursion: _modules
        mods = obj.get("_modules", {})
        for k, v in (mods.items() if isinstance(mods, dict) else []):
            child_full = prefix + ("." if prefix else "") + k
            child_sd = _walk_stub_sd(v, child_full + ".")
            result.update(child_sd)

        # Breadcrumb: "model" key often wraps the real nn.Sequential
        if "model" in obj and obj["model"] is not obj and not isinstance(obj["model"], (int, float, str, bool)):
            result.update(_walk_stub_sd(obj["model"], prefix))

    return result


def _unpack_sd(obj) -> dict[str, torch.Tensor]:
    """From a loaded checkpoint object, extract the state_dict."""
    if isinstance(obj, dict):
        if "model" in obj:
            m = obj["model"]
            if hasattr(m, "state_dict"):
                return m.state_dict()
            elif isinstance(m, dict):
                return {k: v for k, v in m.items() if isinstance(v, torch.Tensor)}
        if "state_dict" in obj:
            return obj["state_dict"]
        if "ema" in obj:
            ema = obj["ema"]
            if hasattr(ema, "state_dict"):
                return ema.float().state_dict()
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")


# ── Entry point ──────────────────────────────────────────────────────────────

def download_weights(variant: str, cache_dir: str | None = None) -> str:
    """Download official pretrained .pt weights for a YOLO variant.

    Parameters
    ----------
    variant : str, e.g. ``"yolov11n"``
    cache_dir : str | None
        Directory to store downloaded weights. Defaults to ``/data/hdd3/chuqi/pretrained/``.

    Returns
    -------
    Local file path to the downloaded .pt file.
    """
    if variant not in _WEIGHT_URLS:
        raise KeyError(
            f"No official weights for '{variant}'. Available: {sorted(_WEIGHT_URLS)}"
        )
    url = _WEIGHT_URLS[variant]
    if cache_dir is None:
        cache_dir = "/data/hdd3/chuqi/pretrained"
    os.makedirs(cache_dir, exist_ok=True)

    fname = os.path.join(cache_dir, f"{variant}.pt")
    if os.path.exists(fname):
        logger.info(f"Weights already cached: {fname}")
        return fname

    logger.info(f"Downloading {variant} weights from {url} ...")
    urllib.request.urlretrieve(url, fname)
    logger.info(f"Downloaded → {fname}")
    return fname


def load_weights(
    model: YOLODetector,
    checkpoint_path: str,
    variant_name: str | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> tuple[int, int, int]:
    """Load pretrained weights into an ElysiaCV YOLO model.

    Supports:
    - Raw PyTorch .pth state_dict (easiest)
    - Ultralytics .pt checkpoint (auto‑handles ultralytics‑only pickle classes)

    Three‑pass matching:
    1. Direct key remap (stem, SPPF, head)
    2. Shape‑based sequential matching (backbone stages, PAN)
    3. Fallback: shape‑only matching (remaining params)
    """
    sd = _extract_state_dict(checkpoint_path)

    # Filter tensors only, strip prefix
    ult_state: list[tuple[str, torch.Tensor]] = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k[6:] if k.startswith("model.") else k
        ult_state.append((nk, v))

    our_shapes = _collect_our_params(model)
    variant = get_variant(variant_name or "yolov11n")

    # ── Pass 1: Direct remap ──
    remapped: dict[str, torch.Tensor] = {}
    remaining: list[tuple[str, torch.Tensor]] = []
    for ult_k, v in ult_state:
        new_k = _remap_direct(ult_k, variant)
        if new_k is not None and new_k in our_shapes and our_shapes[new_k] == v.shape:
            remapped[new_k] = v
        else:
            remaining.append((ult_k, v))

    # ── Pass 2: Sequential shape match ──
    remaining2 = _sequential_match(remaining, our_shapes, variant.depth)
    for ult_k, our_k in remaining2.items():
        remapped[our_k] = dict(remaining)[ult_k]
    remaining3 = [(k, v) for k, v in remaining if k not in remaining2]

    # ── Pass 3: Shape‑only fallback ──
    used_shapes: set[str] = set(remapped.keys())
    by_shape_ours: dict[tuple, list[str]] = {}
    for n, s in our_shapes.items():
        by_shape_ours.setdefault(tuple(s), []).append(n)
    for ult_k, v in remaining3:
        vs = tuple(v.shape)
        if vs in by_shape_ours:
            cands = [n for n in by_shape_ours[vs] if n not in used_shapes]
            if len(cands) == 1:
                remapped[cands[0]] = v
                used_shapes.add(cands[0])
            elif len(cands) > 1:
                for c in cands:
                    if _tail_match(ult_k, c):
                        remapped[c] = v
                        used_shapes.add(c)
                        break

    # ── Load ──
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    loaded = len(remapped)

    if verbose:
        logger.info(
            f"Loaded {loaded}/{len(our_shapes)} params "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    if strict and (missing or unexpected):
        raise RuntimeError(f"Strict load failed: {len(missing)} missing, {len(unexpected)} unexpected")

    return loaded, len(missing), len(unexpected)


def list_available_weights() -> list[str]:
    """Return variants with downloadable official weights."""
    return sorted(_WEIGHT_URLS)