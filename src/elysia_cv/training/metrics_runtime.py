"""Training metrics utilities — adapted from ElysiaRobot metrics_runtime.py.

Key patterns:
- Accumulate micro-batch metrics on device, only detach at log boundary
- gradient_norm: compute only when requested (not every step)
- loss_to_float: boundary conversion from device tensor to Python float
"""

from __future__ import annotations

import math
from typing import Any


def loss_to_float(loss: Any) -> float:
    """Convert loss tensor to Python float (for logging)."""
    if hasattr(loss, "detach"):
        loss = loss.detach()
    if hasattr(loss, "item"):
        return float(loss.item())
    return float(loss)


def detached_metric_value(value: Any) -> Any:
    """Detach a metric value without CPU sync — keeps tensor on device."""
    if hasattr(value, "detach"):
        detached = value.detach()
        if getattr(detached, "numel", lambda: 0)() == 1:
            return detached
        return None
    return value


def gradient_norm(model: Any) -> float | None:
    """Compute global L2 gradient norm. Returns None if no gradients."""
    from elysia_cv.training.distributed import unwrap_model

    total = 0.0
    has_grad = False
    for param in unwrap_model(model).parameters():
        if param.grad is None:
            continue
        has_grad = True
        total += float(param.grad.detach().data.norm(2).item()) ** 2
    return math.sqrt(total) if has_grad else None


def accumulate_micro_metrics(
    accumulator: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Accumulate per-step scalar metrics within one optimizer step.

    Keeps values on device; division happens at ``finalize_micro_metrics``.
    """
    for key, value in metrics.items():
        scalar = _detached_scalar(value)
        if scalar is None:
            last = detached_metric_value(value)
            if last is not None:
                accumulator[str(key)] = {"last": last}
            continue
        record = accumulator.get(str(key))
        if record is None or "sum" not in record:
            accumulator[str(key)] = {"sum": scalar, "count": 1}
            continue
        record["sum"] = _add_scalars(record["sum"], scalar)
        record["count"] = int(record["count"]) + 1


def finalize_micro_metrics(
    accumulator: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Average accumulated micro-batch metrics."""
    finalized: dict[str, Any] = {}
    for key, record in accumulator.items():
        if "sum" not in record:
            finalized[key] = record.get("last")
            continue
        finalized[key] = _divide_scalar(record["sum"], int(record["count"]))
    return finalized


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _detached_scalar(value: Any) -> Any | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "detach"):
        tensor = value.detach()
        if getattr(tensor, "numel", lambda: 0)() == 1:
            return tensor
    return None


def _add_scalars(left: Any, right: Any) -> Any:
    if hasattr(left, "detach") or hasattr(right, "detach"):
        import torch
        if not hasattr(left, "detach"):
            left = torch.tensor(float(left), dtype=getattr(right, "dtype", torch.float32),
                                device=getattr(right, "device", None))
        if not hasattr(right, "detach"):
            right = torch.tensor(float(right), dtype=getattr(left, "dtype", torch.float32),
                                 device=getattr(left, "device", None))
        return left + right.to(device=left.device, dtype=left.dtype)
    return float(left) + float(right)


def _divide_scalar(value: Any, count: int) -> Any:
    if hasattr(value, "detach"):
        return value / max(count, 1)
    return float(value) / max(count, 1)