"""ADE20K Semantic Segmentation Dataset.

ADE20K is a dense-annotation dataset with 150 semantic classes.
Each image has a corresponding ``*_seg.png`` label map where pixel
values encode class indices (0-149).

Expected directory layout::

    root/
      images/
        training/          # .jpg images
        validation/        # .jpg images
      annotations/
        training/          # *_seg.png label maps (0-indexed)
        validation/        # *_seg.png label maps (0-indexed)

Each sample is::

    {
        "inputs":         torch.Tensor [3, H, W],   # after transform pipeline
        "gt_semantic_seg": torch.Tensor [H, W],      # long, class indices
        "ori_shape":      (H, W, 3),                # original shape before transforms
    }

Args:
    root: Root directory (e.g. ``/data/ADEChallengeData2016``).
    split: ``"training"`` or ``"validation"``.
    transforms: List of transform config dicts for the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from elysia_cv.data.transforms import build_pipeline


class ADE20KDataset:
    """ADE20K semantic segmentation dataset (150 classes)."""

    NUM_CLASSES = 150

    def __init__(
        self,
        root: str = "",
        split: str = "training",
        transforms: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms_cfg = transforms or []
        self._pipeline = build_pipeline(list(self.transforms_cfg))

        self._img_dir = self.root / "images" / split
        self._ann_dir = self.root / "annotations" / split

        if not self._img_dir.is_dir():
            raise FileNotFoundError(
                f"ADE20K image directory not found: {self._img_dir}"
            )

        self._samples: list[str] = sorted(
            f.stem for f in self._img_dir.glob("*.jpg")
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        stem = self._samples[idx]
        img_path = str(self._img_dir / f"{stem}.jpg")
        ann_path = str(self._ann_dir / f"{stem}_seg.png")

        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            ori_shape = (img.height, img.width, 3)

        results: dict[str, Any] = {
            "img_path": img_path,
            "ori_shape": ori_shape,
        }

        # Load label map
        if os.path.exists(ann_path):
            seg_map = np.array(Image.open(ann_path), dtype=np.int64)
            seg_map[seg_map >= self.NUM_CLASSES] = 0  # clamp to valid range
            results["gt_semantic_seg"] = seg_map
        else:
            results["gt_semantic_seg"] = np.zeros(
                (ori_shape[0], ori_shape[1]), dtype=np.int64
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
        """ADE20K 150 class names (abbreviated)."""
        return _ADE20K_CLASSES


_ADE20K_CLASSES: list[str] = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth",
    "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea",
    "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers",
    "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway",
    "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet",
    "flower", "book", "hill", "bench", "countertop", "stove",
    "palm", "kitchen island", "computer", "swivel chair", "boat",
    "bar", "arcade machine", "hovel", "bus", "towel", "light",
    "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle",
    "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike",
    "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag",
]