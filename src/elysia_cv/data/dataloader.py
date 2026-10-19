"""DDP-aware DataLoader builder — with ElysiaRobot-validated defaults.

Key optimizations from ElysiaRobot:
- ``num_workers=2``, ``prefetch_factor=2`` (8-GPU tested sweet spot)
- ``pin_memory=True``, ``non_blocking=True`` for H2D copies
- ``persistent_workers=True`` avoids worker re-spawn overhead
- ``multiprocessing_context="spawn"`` when CUDA is available (avoids fork deadlock)
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from elysia_cv.data.sampler import DistributedSampler


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    shuffle: bool = True,
    collate_fn: Any = None,
    distributed: dict[str, Any] | None = None,
    seed: int = 42,
    **extra_kwargs: Any,
) -> DataLoader:
    """Build a DataLoader with ElysiaRobot-validated defaults.

    Args:
        dataset: Map-style dataset.
        batch_size: Per-GPU batch size.
        num_workers: Worker processes (default 2 — ElysiaRobot's sweet spot).
        prefetch_factor: Batches pre-fetched per worker.
        pin_memory: Enable CUDA pinned memory.
        persistent_workers: Keep workers alive across epochs.
        drop_last: Drop incomplete last batch.
        shuffle: Shuffle data (auto-off for distributed samplers).
        distributed: Optional dist config dict with ``enabled`` and ``rank``/``world_size``.
        seed: Random seed.
        **extra_kwargs: Passed to DataLoader constructor.

    Returns:
        Configured DataLoader.
    """
    dist_enabled = (distributed or {}).get("enabled", False)
    sampler = None
    shuffle_effective = shuffle

    if dist_enabled and (distributed or {}).get("world_size", 1) > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=(distributed or {}).get("world_size", 1),
            rank=(distributed or {}).get("rank", 0),
            shuffle=shuffle,
            seed=seed,
        )
        shuffle_effective = False  # sampler handles shuffling

    # Detect CUDA for spawn context (avoid fork-after-CUDA deadlock)
    mp_context = None
    try:
        import torch
        if torch.cuda.is_available() and num_workers > 0:
            import multiprocessing as mp
            mp_context = mp.get_context("spawn")
    except Exception:
        pass

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "drop_last": drop_last,
        "shuffle": shuffle_effective,
        "sampler": sampler,
        "collate_fn": collate_fn,
    }
    if mp_context is not None:
        kwargs["multiprocessing_context"] = mp_context
    kwargs.update(extra_kwargs)

    return DataLoader(dataset, **kwargs)