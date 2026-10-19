"""Distributed training runtime — adapted from ElysiaRobot distributed_runtime.py.

Handles DDP init, model wrapping, gradient sync, metric reduction, and
CUDA backend configuration.  All env-vars use ELYSIACV_* prefix.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Distributed context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributedContext:
    """Distributed runtime context for the current process."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def distributed_context_from_env() -> DistributedContext:
    """Read torchrun / accelerate env vars."""
    world_size = int(os.environ.get("WORLD_SIZE") or "1")
    rank = int(os.environ.get("RANK") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def init_distributed_from_env() -> DistributedContext:
    """Initialize the process group and bind GPU."""
    ctx = distributed_context_from_env()
    if not ctx.enabled:
        return ctx
    if torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank)
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed not available but WORLD_SIZE > 1")
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout = timedelta(seconds=int(os.environ.get("ELYSIACV_DDP_TIMEOUT_SEC") or "3600"))
        torch.distributed.init_process_group(backend=backend, timeout=timeout)
    return ctx


def destroy_distributed(ctx: DistributedContext) -> None:
    """Clean up process group."""
    if ctx.enabled and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# CUDA backend config — ElysiaRobot validated
# ---------------------------------------------------------------------------


def configure_cuda_backend() -> None:
    """Enable cudnn.benchmark, TF32 for matmuls."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# DDP wrapping
# ---------------------------------------------------------------------------


def wrap_ddp(
    model: torch.nn.Module,
    ctx: DistributedContext,
    *,
    find_unused_parameters: bool = True,
    broadcast_buffers: bool = False,
    bucket_cap_mb: int = 64,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
) -> torch.nn.Module:
    """Wrap model with DDP using ElysiaRobot-validated settings."""
    if not ctx.enabled:
        return model
    device_ids = [ctx.local_rank] if torch.cuda.is_available() else None
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return raw module from DDP / FSDP wrapper."""
    # FSDP: model._fsdp_wrapped_module
    if hasattr(model, "_fsdp_wrapped_module"):
        return model._fsdp_wrapped_module
    # DDP: model.module
    return getattr(model, "module", model)


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------


def wrap_fsdp(
    model: torch.nn.Module,
    ctx: DistributedContext,
    *,
    mixed_precision: bool = True,
) -> torch.nn.Module:
    """Wrap model with FSDP (Fully Sharded Data Parallel).

    Uses ``size_based_auto_wrap_policy`` for automatic sharding granularity.
    Falls back to DDP if FSDP is unavailable (PyTorch < 2.0).
    """
    if not ctx.enabled:
        return model

    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        mp = None
        if mixed_precision and torch.cuda.is_available():
            mp = MixedPrecision(
                param_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

        wrapped = FSDP(
            model,
            auto_wrap_policy=size_based_auto_wrap_policy,
            mixed_precision=mp,
            device_id=ctx.local_rank,
        )
        return wrapped

    except ImportError:
        # FSDP not available — fall back to DDP
        return wrap_ddp(model, ctx)


def fsdp_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    """Get full (un-sharded) state dict from an FSDP-wrapped model.

    Falls back to ``unwarp_model(model).state_dict()`` when FSDP is not
    available or the model is not FSDP-wrapped.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return unwrap_model(model).state_dict()

    if not isinstance(model, FSDP):
        return unwrap_model(model).state_dict()

    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        return model.state_dict()


# ---------------------------------------------------------------------------
# Gradient sync control
# ---------------------------------------------------------------------------


def gradient_sync_context(model: torch.nn.Module, ctx: DistributedContext, should_sync: bool):
    """Return no_sync() context when gradient sync should be skipped.

    Used during gradient accumulation: only the last micro-step syncs.
    """
    if ctx.enabled and not should_sync and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


# ---------------------------------------------------------------------------
# Metric reduction — bulk all-reduce pattern from ElysiaRobot
# ---------------------------------------------------------------------------


def reduce_scalar_metrics(
    metrics: dict[str, Any],
    ctx: DistributedContext,
) -> dict[str, Any]:
    """Reduce per-rank scalar metrics via a single batch all-reduce.

    Important: only call at log boundaries (not hot path).  All scalar
    values are collected into a single tensor to avoid N×all_reduce calls.
    """
    if not ctx.enabled or ctx.world_size <= 1:
        return metrics
    reduced = dict(metrics)
    scalar_keys = [k for k, v in metrics.items() if _is_scalar(v)]
    if not scalar_keys:
        return reduced

    device = _reduce_device(metrics)
    values = []
    for key in scalar_keys:
        v = metrics[key]
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(0.0)
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= ctx.world_size
    for i, key in enumerate(scalar_keys):
        reduced[key] = float(tensor[i].item())
    return reduced


def _is_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if hasattr(value, "detach"):
        try:
            return int(value.detach().numel()) == 1
        except Exception:
            return False
    return False


def _reduce_device(metrics: dict[str, Any]) -> torch.device:
    for v in metrics.values():
        if hasattr(v, "detach"):
            t = v.detach()
            if getattr(t, "numel", lambda: 0)() == 1:
                d = getattr(t, "device", None)
                if d is not None and str(d) != "cpu":
                    return torch.device(d)
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}