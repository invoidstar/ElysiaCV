"""Data system — transforms, datasets, dataloaders.

Public API:
  - ``build_pipeline`` — compose transforms from config
  - ``BaseDataset``, ``ImageFolderDataset`` — map-style datasets
  - ``build_dataloader`` — DDP-aware DataLoader builder
  - ``DistributedSampler`` — shard data across ranks
"""

from __future__ import annotations

from elysia_cv.data.transforms import build_pipeline
from elysia_cv.data.dataset import BaseDataset, ImageFolderDataset
from elysia_cv.data.dataloader import build_dataloader
from elysia_cv.data.sampler import DistributedSampler
from elysia_cv.data.cassava_dataset import CassavaDataset

__all__ = [
    "build_pipeline",
    "BaseDataset",
    "ImageFolderDataset",
    "CassavaDataset",
    "build_dataloader",
    "DistributedSampler",
]