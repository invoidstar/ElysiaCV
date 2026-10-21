"""CropAndWeed Dataset — 100-class fine-grained crop/weed detection.

Loads FiftyOne-formatted HuggingFace dataset (Voxel51/CropAndWeed) with
8,034 images, 111,953 bounding boxes, 84 classes (crops + weeds +
growth stages).  Supports simplified class mappings and random
70:15:15 train/val/test split.

Bbox format in samples.json: normalized [x_center, y_center, w, h].
Converted to pixel xyxy on load.

Usage::

    from elysia_cv.data import CropAndWeedDataset

    ds = CropAndWeedDataset("/data/hdd3/chuqi/data/datasets/CropAndWeed",
                            split="train")
    # ds[i] → {"img_path": ..., "gt_bboxes": (N,4) xyxy, "gt_labels": (N,)}
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from elysia_cv.data.dataset import BaseDataset
from elysia_cv.data.transforms import build_pipeline


# ---------------------------------------------------------------------------
# Simplified class mappings (derived from the official CropAndWeed paper)
# ---------------------------------------------------------------------------

# 2-class: crop vs weed
CROP_OR_WEED_2_MAP: dict[str, int] = {
    # Crops → 0
    "Maize": 0,
    "Soybean": 0,
    "Sugar beet": 0,
    "Common bean": 0,
    # Weeds → 1 (all others)
    # 默认: 1
}

# 9-class: 4 crops + 5 weed groups
CROP_OR_WEED_9_MAP: dict[str, int] = {
    "Maize": 0,
    "Sugar beet": 1,
    "Soybean": 2,
    "Common bean": 3,
    # Grass weeds
    "Meadow-grass": 4,
    "Green bristlegrass": 4,
    "Barnyard grass": 4,
    "Cockspur grass": 4,
    "Couch grass": 4,
    "Rye brome": 4,
    # Broadleaf large
    "Field milk thistle": 5,
    "Creeping thistle": 5,
    "Great burdock": 5,
    "Curled dock": 5,
    # Broadleaf small
    "White goosefoot": 6,
    "Red-root amaranth": 6,
    "Small geranium": 6,
    "Knotgrass": 6,
    "Chickweed": 6,
    "Black bindweed": 6,
    "Scentless mayweed": 6,
    "Shepherd's purse": 6,
    "Common fumitory": 6,
    "Cleavers": 6,
    "Common chickweed": 6,
    "Common hemp-nettle": 6,
    "Corn spurry": 6,
    "Field penny-cress": 6,
    "Field pansy": 6,
    "Field forget-me-not": 6,
    "Forget-me-not": 6,
    "Parsley piert": 6,
    "Pale persicaria": 6,
    "Small-flowered crane's-bill": 6,
    "Sun spurge": 6,
    "Scarlet pimpernel": 6,
    # Perennial weeds
    "Hedge bindweed": 7,
    "Field bindweed": 7,
    "Coltsfoot": 7,
    "Yarrow": 7,
    "Canadian thistle": 7,
    # Vegetation (unknown) → 8
    "Vegetation": 8,
}

# Label map name → mapping dict
_SIMPLIFIED_MAPS: dict[str, dict[str, int]] = {
    "crop_or_weed2": CROP_OR_WEED_2_MAP,
    "crop_or_weed9": CROP_OR_WEED_9_MAP,
}

# Crops that define the "crop" group
_CROP_SPECIES = {"Maize", "Sugar beet", "Soybean", "Common bean"}


def _auto_crop_weed2(label: str) -> int:
    """Coarse labeling: 0=crop, 1=weed/vegetation."""
    return 0 if label in _CROP_SPECIES else 1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CropAndWeedDataset(BaseDataset):
    """Fine-grained crop/weed detection dataset.

    Parameters
    ----------
    root : str
        Path to the dataset root containing ``samples.json`` and ``data/``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
    transforms : list[dict] | None
        ElysiaCV transform pipeline config.
    label_mode : str
        ``"full"`` (84 classes), ``"crop_or_weed2"`` (binary), ``"crop_or_weed9"`` (9-class).
    seed : int
        Random seed for split generation.
    """

    def __init__(
        self,
        root: str = "",
        split: str = "train",
        transforms: list[dict[str, Any]] | None = None,
        label_mode: str = "full",
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms_cfg = transforms or []
        self.label_mode = label_mode
        self.seed = seed

        # Load annotations
        self.ann_file = self.root / "samples.json"
        if not self.ann_file.exists():
            raise FileNotFoundError(f"samples.json not found in {self.root}")

        self._class_to_idx: dict[str, int] = {}
        self._classes: list[str] = []

        raw_samples = self._parse_annotations()

        # Split
        self._samples = self._split_samples(raw_samples)

        # Build transforms
        self._transforms = build_pipeline(list(self.transforms_cfg))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_annotations(self) -> list[dict[str, Any]]:
        """Load samples.json and convert to internal format."""
        with open(self.ann_file) as f:
            data = json.load(f)

        raw = data["samples"]
        samples: list[dict[str, Any]] = []
        all_labels: set[str] = set()

        for s in raw:
            filepath = s["filepath"]  # "data/xxx.jpg"
            img_path = str(self.root / filepath)
            if not os.path.exists(img_path):
                continue

            dets = s.get("ground_truth", {}).get("detections", [])
            meta = s.get("metadata", {})
            w, h = meta.get("width", 1920), meta.get("height", 1088)

            bboxes = []
            labels = []

            for det in dets:
                label = det["label"]
                bx, by, bw, bh = det["bounding_box"]
                # Normalized xywh → pixel xyxy
                x1 = bx * w
                y1 = by * h
                x2 = (bx + bw) * w
                y2 = (by + bh) * h
                bboxes.append([x1, y1, x2, y2])
                labels.append(label)
                all_labels.add(label)

            samples.append({
                "img_path": img_path,
                "gt_bboxes": np.array(bboxes, dtype=np.float32) if bboxes
                              else np.zeros((0, 4), dtype=np.float32),
                "gt_labels_raw": labels,  # string labels before remapping
                "ori_shape": (h, w, 3),
                "sample_id": s.get("image_id", ""),
                "moisture": s.get("moisture", ""),
                "lighting": s.get("lighting", ""),
                "separability": s.get("separability", ""),
            })

        # Build class mapping
        self._build_class_map(all_labels)

        # Apply label remapping (to int indices) after class map is built
        for smp in samples:
            raw = smp.pop("gt_labels_raw")
            if self.label_mode != "full":
                remapped = [self._label_remap.get(l, 0) for l in raw]
            else:
                remapped = [self._class_to_idx[l] for l in raw]
            smp["gt_labels"] = np.array(remapped, dtype=np.int64)

        return samples

    def _build_class_map(self, all_labels: set[str]) -> None:
        """Create class_to_idx based on label_mode."""
        if self.label_mode == "full":
            self._classes = sorted(all_labels)
            self._class_to_idx = {c: i for i, c in enumerate(self._classes)}
        elif self.label_mode in _SIMPLIFIED_MAPS:
            remap = _SIMPLIFIED_MAPS[self.label_mode]
            idx_set = set()
            self._label_remap = {}
            for l in all_labels:
                ridx = remap.get(l, 1 if self.label_mode == "crop_or_weed2" else 8)
                self._label_remap[l] = ridx
                idx_set.add(ridx)

            if self.label_mode == "crop_or_weed2":
                group_names = {0: "Crop", 1: "Weed"}
            else:
                group_names = {
                    0: "Maize", 1: "Sugar beet", 2: "Soybean", 3: "Common bean",
                    4: "Grass weeds", 5: "Broadleaf large",
                    6: "Broadleaf small", 7: "Perennial weeds",
                    8: "Vegetation/Unknown",
                }
            self._classes = [group_names[i] for i in sorted(idx_set)]
            self._class_to_idx = {name: i for i, name in enumerate(self._classes)}
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode!r}")

    def _split_samples(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply deterministic random 70:15:15 split."""
        if self.split == "all":
            return samples

        rng = random.Random(self.seed)
        indices = list(range(len(samples)))
        rng.shuffle(indices)

        n = len(samples)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        # n_test = n - n_train - n_val

        splits = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }

        if self.split not in splits:
            raise ValueError(f"Unknown split: {self.split!r}")
        return [samples[i] for i in splits[self.split]]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index].copy()
        sample = self._load_image(sample)
        for transform in self._transforms:
            sample = transform(sample)
        return sample

    def _load_image(self, sample: dict[str, Any]) -> dict[str, Any]:
        return sample

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return dict(self._class_to_idx)


