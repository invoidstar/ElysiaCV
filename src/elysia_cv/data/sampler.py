"""DistributedSampler — shard dataset across DDP ranks."""

from __future__ import annotations

import math
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