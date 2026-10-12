"""BaseDataset — Map-style PyTorch Dataset with composable transforms.

Usage::

    dataset = BaseDataset(
        root="/data/imagenet",
        split="train",
        transforms=[{"type": "Resize", "size": 224}, {"type": "PackInputs"}],
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from elysia_cv.data.transforms import build_pipeline


class BaseDataset:
    """Map-style dataset with composable transforms.

    Subclasses can override ``_load_annotations`` and ``_load_image``
    to adapt to different dataset formats.
    """

    def __init__(
        self,
        root: str = "",
        split: str = "train",
        transforms: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms_cfg = transforms or []
        self._transforms = build_pipeline(list(self.transforms_cfg))
        self._samples: list[dict[str, Any]] = []
        if self.root.is_dir():
            self._samples = self._load_annotations()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index].copy()
        sample = self._load_image(sample)
        for transform in self._transforms:
            sample = transform(sample)
        return sample

    # ------------------------------------------------------------------
    # Override hooks for subclasses
    # ------------------------------------------------------------------

    def _load_annotations(self) -> list[dict[str, Any]]:
        """Discover image paths. Override for custom dataset formats."""
        samples: list[dict[str, Any]] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for path in self.root.rglob(ext):
                samples.append({"img_path": str(path)})
        return sorted(samples, key=lambda s: s["img_path"])

    def _load_image(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Optionally pre-load image if path is set. Default: defer to transforms."""
        return sample


class ImageFolderDataset(BaseDataset):
    """ImageFolder variant — expects ``root/label_name/image.jpg`` layout."""

    def __init__(
        self,
        root: str = "",
        split: str = "train",
        transforms: list[dict[str, Any]] | None = None,
    ) -> None:
        self._classes: list[str] = []
        self._class_to_idx: dict[str, int] = {}
        super().__init__(root=root, split=split, transforms=transforms)

    def _load_annotations(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        entries = sorted(
            d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        self._classes = [d.name for d in entries]
        self._class_to_idx = {name: i for i, name in enumerate(self._classes)}
        for class_dir in entries:
            class_idx = self._class_to_idx[class_dir.name]
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for img_path in sorted(class_dir.glob(ext)):
                    samples.append({
                        "img_path": str(img_path),
                        "gt_label": class_idx,
                    })
        return samples

    @property
    def num_classes(self) -> int:
        return len(self._classes)