#!/usr/bin/env python3
"""Model profiler: FLOPs estimation + parameter count per layer.

Supports Conv2d, Linear, BatchNorm2d, LayerNorm, MaxPool2d,
AvgPool2d, AdaptiveAvgPool2d, ReLU / GELU / Sigmoid, and Dropout.

Usage:
    python tools/profile.py configs/experiments/cassava_resnet50.yaml
    python tools/profile.py <config.yaml> --input-size 3 224 224
    python tools/profile.py <config.yaml> --verbose   # per-layer detail

Output: prints a summary table to stdout (FLOPs, #params, ratio per layer).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn


# ── FLOPs registry ──
# Each hook: (module, input_tensor, output_tensor) → (flops: int, params: int)

_FLOP_FNS: dict[type, Callable] = {}


def _register(*types: type):
    def dec(fn):
        for t in types:
            _FLOP_FNS[t] = fn
        return fn
    return dec


@_register(nn.Conv2d)
def _conv2d_flops(m: nn.Conv2d, inp, out) -> tuple[int, int]:
    """Conv2d FLOPs: 2 * k_h * k_w * C_in * C_out * H_out * W_out (MACs ×2 for FMA)."""
    if hasattr(m, "weight") and m.weight is not None:
        params = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    else:
        params = 0
    if len(inp[0].shape) != 4:
        return 0, params
    _, cin, hin, win = inp[0].shape
    _, cout, hout, wout = out.shape
    groups = m.groups or 1
    kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (cin // groups)
    total_ops = kernel_ops * cout * hout * wout * 2  # MAC × 2
    return total_ops, params


@_register(nn.Linear)
def _linear_flops(m: nn.Linear, inp, out) -> tuple[int, int]:
    params = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    batch = inp[0].shape[0] if len(inp[0].shape) >= 2 else 1
    flops = 2 * m.in_features * m.out_features * batch  # MAC × 2
    return flops, params


@_register(nn.BatchNorm2d)
def _batchnorm_flops(m: nn.BatchNorm2d, inp, out) -> tuple[int, int]:
    params = 2 * m.num_features  # weight + bias
    if len(inp[0].shape) == 4:
        flops = 2 * m.num_features * out.shape[2] * out.shape[3]
    else:
        flops = 0
    return flops, params


@_register(nn.LayerNorm)
def _layernorm_flops(m: nn.LayerNorm, inp, out) -> tuple[int, int]:
    params = out.numel() * 2  # weight + bias per element
    flops = params  # subtract mean, divide std
    return flops, params


@_register(nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)
def _pool_flops(m, inp, out) -> tuple[int, int]:
    if len(out.shape) == 4:
        flops = out.shape[1] * out.shape[2] * out.shape[3]
    else:
        flops = 0
    return flops, 0


@_register(nn.ReLU, nn.ReLU6, nn.GELU, nn.Sigmoid, nn.Tanh)
def _act_flops(m, inp, out) -> tuple[int, int]:
    flops = out.numel()
    return flops, 0


@_register(nn.Dropout)
def _dropout_flops(m, inp, out) -> tuple[int, int]:
    return 0, 0


# ── Profiler core ──

class Profiler:
    """Register forward hooks and collect per-layer FLOPs + param counts."""

    def __init__(self, model: nn.Module):
        self._model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._stats: list[dict[str, Any]] = []

    def _hook(self, name: str) -> Callable:
        def fn(module: nn.Module, inp, out):
            total_flops = 0
            total_params = 0
            matched = False
            for cls in _FLOP_FNS:
                if isinstance(module, cls):
                    f, p = _FLOP_FNS[cls](module, inp, out)
                    total_flops += f
                    total_params += p
                    matched = True
                    break
            if not matched:
                total_params = sum(p.numel() for p in module.parameters(recurse=False))
            self._stats.append({
                "name": name,
                "type": module.__class__.__name__,
                "flops": total_flops,
                "params": total_params,
            })
        return fn

    def register_hooks(self) -> None:
        for name, module in self._model.named_modules():
            # Skip containers (Sequential, ModuleList, the root module)
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                continue
            if len(list(module.children())) == 0 or isinstance(
                module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm,
                         nn.ReLU, nn.GELU, nn.Dropout, nn.MaxPool2d, nn.AvgPool2d)
            ):
                self._handles.append(module.register_forward_hook(self._hook(name)))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def profile(self, input_tensor: torch.Tensor) -> list[dict[str, Any]]:
        self._stats.clear()
        self.register_hooks()
        try:
            self._model.eval()
            with torch.no_grad():
                self._model(input_tensor)
        finally:
            self.remove_hooks()
        return self._stats


# ── Report ──

def format_number(x: int) -> str:
    """Format: 1,234,567 → 1.23 M"""
    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f} G"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f} M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.2f} K"
    return str(x)


def print_profile_report(stats: list[dict[str, Any]], *, verbose: bool = False) -> None:
    total_flops = sum(s["flops"] for s in stats)
    total_params = sum(s["params"] for s in stats)

    # Group by top-level prefix
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"flops": 0, "params": 0})
    for s in stats:
        prefix = s["name"].split(".")[0] if "." in s["name"] else s["name"]
        groups[prefix]["flops"] += s["flops"]
        groups[prefix]["params"] += s["params"]

    # Summary table
    print()
    print("=" * 78)
    print(f"{'Module':<30s} {'FLOPs':>14s} {'Params':>14s}  {'%FLOPs':>7s} {'%Params':>7s}")
    print("=" * 78)

    for prefix in sorted(groups):
        g = groups[prefix]
        flop_pct = g["flops"] / max(total_flops, 1) * 100
        param_pct = g["params"] / max(total_params, 1) * 100
        print(f"{prefix:<30s} {format_number(g['flops']):>14s} {format_number(g['params']):>14s}  {flop_pct:6.2f}% {param_pct:6.2f}%")

    print("-" * 78)
    print(f"{'TOTAL':<30s} {format_number(total_flops):>14s} {format_number(total_params):>14s}  {'100.00%':>7s} {'100.00%':>7s}")
    print("=" * 78)

    # Per-layer detail (verbose)
    if verbose and stats:
        print()
        print(f"{'Layer':<50s} {'Type':<20s} {'FLOPs':>12s} {'Params':>12s}")
        print("-" * 96)
        for s in stats:
            if s["flops"] == 0 and s["params"] == 0:
                continue
            print(f"{s['name']:<50s} {s['type']:<20s} {format_number(s['flops']):>12s} {format_number(s['params']):>12s}")

        print()

    # High-level summary for logging
    print(f"  Total FLOPs : {format_number(total_flops)}")
    print(f"  Total Params: {format_number(total_params)}  ({format_number(total_params * 4)} fp32 bytes)")

    if _captured_model is not None:
        trainable = sum(p.numel() for p in _captured_model.parameters() if p.requires_grad)
    else:
        trainable = 0
    print(f"  Trainable   : {format_number(trainable)}")
    print()


_captured_model: nn.Module | None = None


def profile_model(model: nn.Module, input_shape: tuple[int, ...], *,
                  device: torch.device | None = None,
                  verbose: bool = False) -> None:
    """Run FLOPs + parameter count on a model."""
    global _captured_model
    _captured_model = model

    if device is None:
        device = next(model.parameters()).device

    dummy = torch.randn(*input_shape, device=device)
    profiler = Profiler(model)
    stats = profiler.profile(dummy)
    print_profile_report(stats, verbose=verbose)


# ── CLI ──

def main() -> None:
    parser = argparse.ArgumentParser(description="Model FLOPs + parameter profiler")
    parser.add_argument("config", help="path to YAML config file")
    parser.add_argument("--input-size", nargs=3, type=int, default=[3, 224, 224],
                        metavar=("C", "H", "W"), help="input shape (default: 3 224 224)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show per-layer breakdown")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from elysia_cv.config import load_cv_config_with_object
    from elysia_cv.models.classification.cassava.runtime import build_model as build_cassava
    from elysia_cv.models.classification.resnet.runtime import build_model as build_resnet

    # Load config
    print(f"Loading config: {args.config}")
    cfg = load_cv_config_with_object(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model_type = cfg.model.type
    model_cfg = {
        "model": {
            "type": model_type,
            "backbone": dict(cfg.model.backbone),
            "head": dict(cfg.model.head),
        }
    }

    if model_type == "cassava":
        model = build_cassava(model_cfg)
    else:
        model = build_resnet(model_cfg)

    model = model.to(device)

    input_shape = (1, *args.input_size)
    print(f"Input shape: {input_shape}")
    profile_model(model, input_shape, device=device, verbose=args.verbose)


if __name__ == "__main__":
    main()