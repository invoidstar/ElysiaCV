"""DistributedSampler — shard dataset across DDP ranks."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch
from torch.utils.data import Sampler, Dataset


class DistributedSampler(Sampler):
    """Samples a subset of the dataset per DDP rank.

    Compatible with torchrun / accelerate launchers.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Deterministic shuffle per epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = indices

        # Pad to make evenly divisible
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[: self.total_size]

        # Subsample for this rank
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling deterministically across epochs."""
        self.epoch = epoch


class ClassBalancedSampler(Sampler):
    """Sample indices such that each class appears with equal probability.

    Useful when the dataset has severe class imbalance (e.g. Cassava
    where some diseases are 10× rarer than others).

    Args:
        dataset: A map-style dataset whose ``__getitem__`` returns
            ``{"gt_label": int}`` or ``(img, label)`` tuples.
            If the dataset has a ``.targets`` or ``.labels`` attribute,
            that is used directly.
        num_samples: Number of samples to draw per epoch (default: len(dataset)).
        shuffle: Shuffle within each class.
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Discover class labels
        self._labels = _extract_labels(dataset)
        self._class_indices = _build_class_indices(self._labels)

        self.num_classes = len(self._class_indices)
        self.samples_per_class = max(
            1, int(math.ceil(self.num_samples / self.num_classes))
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices: list[int] = []
        for cls_idx in sorted(self._class_indices.keys()):
            pool = self._class_indices[cls_idx]
            if self.shuffle:
                perm = torch.randperm(len(pool), generator=g).tolist()
                pool = [pool[p] for p in perm]
            # Repeating sampling to fill samples_per_class
            for j in range(self.samples_per_class):
                indices.append(pool[j % len(pool)])

        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[p] for p in perm]

        return iter(indices[: self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _extract_labels(dataset: Dataset) -> list[int]:
    """Extract integer class labels from a dataset.

    Tries multiple strategies: ``.targets``, ``.labels``, ``.classes``
    attributes, or iterating the first few samples.
    """
    # Strategy 1: direct attribute
    for attr in ("targets", "labels", "classes"):
        val = getattr(dataset, attr, None)
        if val is not None and isinstance(val, (list, torch.Tensor)):
            return [int(v) for v in val]

    # Strategy 2: iterate over the dataset
    try:
        labels: list[int] = []
        for i in range(min(len(dataset), 1000)):
            sample = dataset[i]
            if isinstance(sample, dict) and "gt_label" in sample:
                labels.append(int(sample["gt_label"]))
            elif isinstance(sample, (tuple, list)) and len(sample) == 2:
                labels.append(int(sample[1]))
        if labels:
            return labels + labels * (len(dataset) // len(labels))
    except Exception:
        pass

    # Fallback: uniform
    return [0] * len(dataset)


def _build_class_indices(labels: list[int]) -> dict[int, list[int]]:
    """Build ``{class_id: [index, ...]}`` mapping."""
    class_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(int(label), []).append(idx)
    return class_indices


def compute_class_weights(
    dataset: Dataset,
    *,
    num_classes: int | None = None,
    beta: float = 0.9999,
) -> torch.Tensor:
    """Compute per-class loss weights for imbalanced datasets.

    Uses effective-number re-weighting (Cui et al., CVPR 2019)::

        weight_c = (1 - beta) / (1 - beta^{n_c})

    where n_c is the number of samples in class c.  This smoothly
    transitions from inverse-frequency-like (small β ≪ 1) to
    near-uniform (β → 1).

    Special case: ``beta=0.0`` uses pure inverse-frequency
    ``weight_c = total_samples / (num_classes * n_c)``.

    Args:
        dataset: Dataset with labels.
        num_classes: Total number of classes (auto-detect if None).
        beta: Smoothing factor (default 0.9999).

    Returns:
        ``tensor([w_0, w_1, ..., w_{C-1}])`` of per-class weights,
        normalized so mean weight = 1.
    """
    labels = _extract_labels(dataset)
    counter = Counter(labels)

    if num_classes is None:
        num_classes = max(counter.keys()) + 1

    counts = [counter.get(c, 0) for c in range(num_classes)]

    weights = []
    if beta <= 0.0:
        # Pure inverse-frequency
        total = max(sum(counts), 1)
        for n_c in counts:
            w = total / (num_classes * max(n_c, 1))
            weights.append(w)
    else:
        for n_c in counts:
            if n_c == 0:
                w = 0.0
            else:
                w = (1.0 - beta) / (1.0 - beta ** n_c)
            weights.append(w)

    w_t = torch.tensor(weights, dtype=torch.float32)
    # Normalize so mean weight = 1
    mean_w = w_t[w_t > 0].mean()
    if mean_w > 0:
        w_t = w_t / mean_w
    return w_t