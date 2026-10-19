"""Cityscapes Semantic Segmentation Dataset.

Cityscapes is a urban-scene understanding dataset with 19 semantic classes
(trainId).  Each image is accompanied by a ``*_gtFine_labelIds`` PNG and
a ``*_gtFine_polygons.json`` with instance-level annotations.  This dataset
maps ``labelIds`` → ``trainId`` (19 classes + ignore).

Expected directory layout::

    root/
      leftImg8bit/
        train/
          {city}/
            *{id}_leftImg8bit.png
        val/
        test/
      gtFine/
        train/
          {city}/
            *{id}_gtFine_labelIds.png
            *{id}_gtFine_polygons.json
        val/
        test/

Each sample is::

    {
        "inputs":          torch.Tensor [3, H, W],
        "gt_semantic_seg": torch.Tensor [H, W],   # trainId: 0-18, 255=ignore
        "ori_shape":       (H, W, 3),
    }

Args:
    root: Root directory (e.g. ``/data/cityscapes``).
    split: ``"train"``, ``"val"``, or ``"test"``.
    transforms: List of transform config dicts for the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from elysia_cv.data.transforms import build_pipeline


class CityscapesDataset:
    """Cityscapes semantic segmentation dataset (19 classes)."""

    NUM_CLASSES = 19
    IGNORE_ID = 255

    # Mapping from labelId (0-33) → trainId (0-18, 255=ignore)
    # Source: https://github.com/mcordts/cityscapesScripts/blob/master/
    #         cityscapesscripts/helpers/labels.py
    _LABEL_TO_TRAIN: dict[int, int] = {
        0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
        7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
        14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
        -1: 255,
    }

    def __init__(
        self,
        root: str = "",
        split: str = "train",
        transforms: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms_cfg = transforms or []
        self._pipeline = build_pipeline(list(self.transforms_cfg))

        self._img_dir = self.root / "leftImg8bit" / split
        self._ann_dir = self.root / "gtFine" / split

        if not self._img_dir.is_dir():
            raise FileNotFoundError(
                f"Cityscapes image directory not found: {self._img_dir}"
            )

        self._samples: list[tuple[str, str]] = self._scan_samples()

    def _scan_samples(self) -> list[tuple[str, str]]:
        samples: list[tuple[str, str]] = []
        for city_dir in sorted(self._img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = img_path.stem.replace("_leftImg8bit", "")
                city = city_dir.name
                samples.append((city, stem))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        city, stem = self._samples[idx]
        img_path = str(
            self._img_dir / city / f"{stem}_leftImg8bit.png"
        )
        ann_path = str(
            self._ann_dir / city / f"{stem}_gtFine_labelIds.png"
        )

        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            ori_shape = (img.height, img.width, 3)

        results: dict[str, Any] = {
            "img_path": img_path,
            "ori_shape": ori_shape,
        }

        # Load label map and convert labelIds → trainId
        if os.path.exists(ann_path):
            label_ids = np.array(Image.open(ann_path), dtype=np.int64)
            train_ids = np.full_like(label_ids, fill_value=self.IGNORE_ID, dtype=np.int64)
            for lid, tid in self._LABEL_TO_TRAIN.items():
                train_ids[label_ids == lid] = tid
            results["gt_semantic_seg"] = train_ids
        else:
            results["gt_semantic_seg"] = np.full(
                (ori_shape[0], ori_shape[1]),
                fill_value=self.IGNORE_ID, dtype=np.int64,
            )

        # Run through transform pipeline
        for transform in self._pipeline:
            results = transform(results)

        return results

    @property
    def num_classes(self) -> int:
        return self.NUM_CLASSES

    @property
    def class_names(self) -> list[str]:
        return _CITYSCAPES_CLASSES


_CITYSCAPES_CLASSES: list[str] = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]