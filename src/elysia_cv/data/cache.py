"""Dataset caching — preload dataset into memory (or LMDB) for fast iteration.

For small/medium datasets (e.g. Cassava ≈21k images), I/O can be the
bottleneck.  ``CachedDataset`` preloads every sample into RAM so that
``__getitem__`` becomes a dictionary lookup instead of a disk read.

Optional LMDB backend for disk-backed caching of larger datasets.

Usage::

    from elysia_cv.data.cache import CachedDataset

    # Memory cache (fastest for datasets < 10 GB)
    ds = CachedDataset(original_dataset, cache_dir=None)

    # LMDB cache (disk-persistent, survives restarts)
    ds = CachedDataset(original_dataset,
                       cache_dir="/tmp/my_cache.lmdb",
                       cache_mode="lmdb")

    dataloader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """Preload a wrapped dataset into memory for fast random access.

    Args:
        dataset: Any map-style dataset (must implement ``__len__`` and
            ``__getitem__``).
        cache_dir: Path to an LMDB or directory for persistent cache.
            Set to ``None`` for in-memory-only (fastest).
        cache_mode: ``"memory"`` (keep tensors in RAM) or ``"lmdb"``
            (disk-backed key-value store, for very large datasets).
        num_workers: Number of worker threads for preloading.

    After construction, ``len(cached) == len(dataset)`` and
    ``cached[i]`` returns the same dict as ``dataset[i]`` without
    running the transform pipeline or reading from disk.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        cache_dir: str | None = None,
        cache_mode: str = "memory",
        num_workers: int = 0,
    ) -> None:
        self._original = dataset
        self._cache_mode = cache_mode
        self._cache: dict[int, dict[str, Any]] = {}
        self._cache_dir: str | None = None

        if cache_mode == "lmdb" and cache_dir is not None:
            self._cache_dir = cache_dir

        # Build data key for cache invalidation
        self._data_key = self._compute_data_key(dataset)

        # Check if cached version exists
        if self._cache_dir and self._cache_valid():
            self._load_from_lmdb()
        else:
            self._preload(num_workers=num_workers)
            if self._cache_dir and cache_mode == "lmdb":
                self._save_to_lmdb()

    # ── Dataset protocol ──

    def __len__(self) -> int:
        return len(self._original)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._cache[idx]

    # ── Preloading ──

    def _preload(self, *, num_workers: int = 0) -> None:
        N = len(self._original)
        # Single-threaded preload (safe across transform pipelines)
        for i in range(N):
            self._cache[i] = _tensorize_sample(self._original[i])
        print(f"CachedDataset: preloaded {N} samples into memory.")

    # ── LMDB persistence ──

    def _compute_data_key(self, dataset: Dataset) -> str:
        """Compute a hash key to detect dataset changes."""
        h = hashlib.sha256()
        h.update(str(len(dataset)).encode())
        # Hash first 3 samples as fingerprint
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            h.update(str(sorted(sample.keys())).encode())
        return h.hexdigest()[:16]

    def _cache_valid(self) -> bool:
        if not self._cache_dir:
            return False
        meta_path = os.path.join(self._cache_dir, "data_key.txt")
        if not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            stored_key = f.read().strip()
        return stored_key == self._data_key

    def _save_to_lmdb(self) -> None:
        assert self._cache_dir is not None
        os.makedirs(self._cache_dir, exist_ok=True)

        # Save metadata
        with open(os.path.join(self._cache_dir, "data_key.txt"), "w") as f:
            f.write(self._data_key)

        # Save samples as individual .pt files (simple, debuggable)
        data_dir = os.path.join(self._cache_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        for i, sample in self._cache.items():
            torch.save(sample, os.path.join(data_dir, f"{i:06d}.pt"))

        # Save length
        torch.save(len(self._cache), os.path.join(self._cache_dir, "len.pt"))
        print(f"CachedDataset: saved {len(self._cache)} samples to {self._cache_dir}")

    def _load_from_lmdb(self) -> None:
        assert self._cache_dir is not None
        data_dir = os.path.join(self._cache_dir, "data")
        N = torch.load(os.path.join(self._cache_dir, "len.pt"))
        for i in range(int(N)):
            path = os.path.join(data_dir, f"{i:06d}.pt")
            if os.path.exists(path):
                self._cache[i] = torch.load(path, weights_only=False)
        print(f"CachedDataset: loaded {len(self._cache)} samples from {self._cache_dir}")


def _tensorize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays in sample to torch tensors if needed."""
    out: dict[str, Any] = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v.copy())
        elif isinstance(v, torch.Tensor):
            out[k] = v.clone().detach()
        elif isinstance(v, (list, tuple)):
            # List of tensors (e.g. bboxes) — clone each
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                out[k] = [e.clone().detach() for e in v]
            else:
                out[k] = v
        else:
            out[k] = v
    return out