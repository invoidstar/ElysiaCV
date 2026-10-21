"""Data system — transforms, datasets, dataloaders.

Public API:
  - ``build_pipeline`` — compose transforms from config
  - ``BaseDataset``, ``ImageFolderDataset`` — map-style datasets
  - ``build_dataloader`` — DDP-aware DataLoader builder
  - ``DistributedSampler`` — shard data across ranks
"""

from __future__ import annotations

from elysia_cv.data.transforms import build_pipeline, build_batch_pipeline, Compose
from elysia_cv.data.dataset import BaseDataset, ImageFolderDataset
from elysia_cv.data.dataloader import build_dataloader
from elysia_cv.data.sampler import DistributedSampler, ClassBalancedSampler, compute_class_weights
from elysia_cv.data.cassava_dataset import CassavaDataset
from elysia_cv.data.coco_dataset import CocoDetectionDataset
from elysia_cv.data.ade20k_dataset import ADE20KDataset
from elysia_cv.data.cityscapes_dataset import CityscapesDataset
from elysia_cv.data.cropandweed_dataset import CropAndWeedDataset
from elysia_cv.data.video_dataset import VideoDataset
from elysia_cv.data.cache import CachedDataset
from elysia_cv.data.prefetcher import DataPrefetcher
from elysia_cv.data.adapters import (
    HFClassificationAdapter,
    TorchVisionClassificationAdapter,
    TorchVisionDetectionAdapter,
)

__all__ = [
    "build_pipeline",
    "Compose",
    "BaseDataset",
    "ImageFolderDataset",
    "CassavaDataset",
    "CocoDetectionDataset",
    "ADE20KDataset",
    "CityscapesDataset",
    "CropAndWeedDataset",
    "VideoDataset",
    "CachedDataset",
    "build_dataloader",
    "DistributedSampler",
    "ClassBalancedSampler",
    "compute_class_weights",
    "DataPrefetcher",
    "HFClassificationAdapter",
    "TorchVisionClassificationAdapter",
    "TorchVisionDetectionAdapter",
]