"""COCO Object Detection Dataset.

Follows MMDetection conventions: pre-loads COCO JSON annotations,
pairs images with annotations, and runs the ElysiaCV transform pipeline
(LoadImageFromFile → LoadAnnotations → Resize → ... → PackDetInputs).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from elysia_cv.data.transforms import build_pipeline


class CocoDetectionDataset:
    """COCO-style object detection dataset.

    Expected directory layout::

        root/
          images/{split}/       # e.g. images/train2017/
          annotations/instances_{split}.json

    Each sample returned is a dict with::

        {
            "inputs":        torch.Tensor [3, H, W],
            "gt_bboxes":     torch.Tensor [N, 4]  (xyxy format),
            "gt_labels":     torch.Tensor [N],
            "image_id":      int,
            "ori_shape":     (H, W, 3),
        }

    Args:
        root: Root directory (e.g. ``/data/coco``).
        split: Split name (e.g. ``"train2017"``).
        transforms: List of transform config dicts for the :func:`build_pipeline`.
    """

    # ------------------------------------------------------------------
    # Class-level COCO → 0-based class mapping (maintained across splits)
    # ------------------------------------------------------------------
    _category_id_to_contiguous: dict[int, int] | None = None
    _contiguous_id_to_category: dict[int, int] | None = None

    def __init__(
        self,
        root: str = "",
        split: str = "train2017",
        transforms: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms_cfg = transforms or []
        self._pipeline = build_pipeline(list(self.transforms_cfg))

        # Locate annotation file
        ann_dir = self.root / "annotations"
        candidates = [
            ann_dir / f"instances_{split}.json",
            ann_dir / f"{split}.json",
            ann_dir / f"instances_{split.replace('2017', '')}2017.json",
        ]
        self.ann_file = None
        for c in candidates:
            if c.exists():
                self.ann_file = c
                break

        if self.ann_file is None:
            raise FileNotFoundError(
                f"No COCO annotation file found for split={split!r} in {ann_dir}. "
                f"Checked: {candidates}"
            )

        # Load COCO JSON
        with open(self.ann_file) as f:
            coco_data = json.load(f)

        # Build category mapping (once per process)
        if CocoDetectionDataset._category_id_to_contiguous is None:
            cat_ids = sorted(cat["id"] for cat in coco_data.get("categories", []))
            CocoDetectionDataset._category_id_to_contiguous = {
                cid: i for i, cid in enumerate(cat_ids)
            }
            CocoDetectionDataset._contiguous_id_to_category = {
                i: cid for i, cid in enumerate(cat_ids)
            }

        self.cat2label = CocoDetectionDataset._category_id_to_contiguous

        # Build image index
        self._images: dict[int, dict] = {
            img["id"]: img for img in coco_data["images"]
        }

        # Build annotation index (image_id → list of annotations)
        img_to_anns: dict[int, list[dict]] = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            img_to_anns[ann["image_id"]].append(ann)

        # Filter to images that have annotations AND exist on disk
        self._sample_ids: list[int] = []
        img_dir = self.root / "images" / self.split
        for img_id, img_info in self._images.items():
            img_path = img_dir / img_info["file_name"]
            if img_path.exists() and img_id in img_to_anns:
                self._sample_ids.append(img_id)

        self._img_to_anns = img_to_anns

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id = self._sample_ids[idx]
        img_info = self._images[img_id]
        anns = self._img_to_anns.get(img_id, [])

        # Build pre-transform results dict (MMDetection convention)
        img_path = str(self.root / "images" / self.split / img_info["file_name"])
        results: dict[str, Any] = {
            "img_path": img_path,
            "img_id": img_id,
            "ori_shape": (img_info["height"], img_info["width"], 3),
            "gt_bboxes": [],
            "gt_labels": [],
        }

        for ann in anns:
            # Skip crowd annotations
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, w, h]
            results["gt_bboxes"].append([x, y, x + w, y + h])  # → xyxy
            results["gt_labels"].append(
                self.cat2label.get(ann["category_id"], 0)
            )

        # Run through transform pipeline
        for transform in self._pipeline:
            results = transform(results)

        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.cat2label)

    @property
    def class_names(self) -> list[str]:
        # COCO categories are only available if JSON has "categories" with "name"
        with open(self.ann_file) as f:
            coco = json.load(f)
        cats = coco.get("categories", [])
        if categories:
            return [c.get("name", str(c["id"])) for c in sorted(cats, key=lambda x: x["id"])]
        return [str(i) for i in range(self.num_classes)]