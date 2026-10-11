"""Environment detection utilities for distributed training."""

from __future__ import annotations

import os


def is_distributed() -> bool:
    """Return True if running under torch.distributed."""
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def get_rank() -> int:
    """Return the global rank, or 0 if not distributed."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def get_world_size() -> int:
    """Return the world size, or 1 if not distributed."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def get_local_rank() -> int:
    """Return the local rank from environment variables."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Return True on rank 0."""
    return get_rank() == 0


def get_device() -> str:
    """Return the current device string ('cuda:N' or 'cpu')."""
    if is_distributed():
        return f"cuda:{get_local_rank()}"
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"