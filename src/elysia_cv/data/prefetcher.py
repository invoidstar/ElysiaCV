"""DataPrefetcher — asynchronous CPU → GPU batch streaming.

Overlaps data loading with compute by pre-fetching the next batch onto the
GPU while the current batch is being processed.  This is the standard
technique used by Detectron2, MMEngine, and timm.
"""

from __future__ import annotations

from typing import Any

import torch


class DataPrefetcher:
    """Asynchronously prefetch batches to a CUDA device.

    Usage::

        loader = DataLoader(dataset, batch_size=8)
        prefetcher = DataPrefetcher(loader, device="cuda:0")

        batch = next(prefetcher)   # Returns the *preloaded* batch (on GPU)
        while batch is not None:
            model(batch)           # Already on GPU — no data transfer stall
            batch = next(prefetcher)
    """

    def __init__(self, loader: Any, device: torch.device | str = "cuda:0") -> None:
        self.loader = loader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        self._iter = iter(loader)
        self._preloaded: Any = None
        self._init()

    def _init(self) -> None:
        """Prefetch the first batch synchronously (no stream overlap yet)."""
        try:
            batch = next(self._iter)
            self._preloaded = self._to_device(batch)
        except StopIteration:
            self._preloaded = None

    def _to_device(self, batch: Any) -> Any:
        """Move a batch to the target device (non-blocking for CUDA)."""
        if batch is None:
            return None
        if isinstance(batch, dict):
            return {
                k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else self._to_device(v)
                for k, v in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        return batch

    def __iter__(self) -> DataPrefetcher:
        return self

    def __next__(self) -> Any:
        if self._preloaded is None:
            raise StopIteration

        current = self._preloaded

        # Prefetch next batch while the caller processes ``current``
        try:
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    next_batch = next(self._iter)
                    self._preloaded = self._to_device(next_batch)
            else:
                self._preloaded = self._to_device(next(self._iter))
        except StopIteration:
            self._preloaded = None

        # Wait for the current batch's copy to finish (in case it hasn't)
        if self.stream is not None:
            self.stream.synchronize()

        return current

    def __len__(self) -> int:
        return len(self.loader)