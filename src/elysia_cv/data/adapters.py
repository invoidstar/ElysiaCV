"""Data adapters — wrap external datasets into ElysiaCV-compatible formats.

Provides lightweight adapters for HuggingFace ``datasets``, torchvision
built-in datasets, and OpenDataLab / custom formats.  Each adapter converts
the source dataset's ``__getitem__`` output into the ``{"inputs": ..., "gt_label": ...}``
dict expected by ElysiaCV transforms and training loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class HFClassificationAdapter:
    """Wrap a HuggingFace ``datasets.Dataset`` for classification.

    Usage::

        from datasets import load_dataset
        ds = load_dataset("cifar10", split="train")
        adapter = HFClassificationAdapter(ds, image_key="img", label_key="label")
        # adapter[i] → {"inputs": PIL.Image, "gt_label": int}
    """

    def __init__(
        self,
        dataset: Any,
        *,
        image_key: str = "image",
        label_key: str = "label",
    ) -> None:
        self._ds = dataset
        self._image_key = image_key
        self._label_key = label_key

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._ds[idx]
        img = sample[self._image_key]
        # HF datasets may return PIL or numpy
        if isinstance(img, (np.ndarray,)):
            img = Image.fromarray(img)
        return {
            "inputs": img,
            "gt_label": int(sample[self._label_key]),
        }


class TorchVisionClassificationAdapter:
    """Wrap a torchvision dataset for classification.

    Usage::

        import torchvision.datasets as tvds
        ds = tvds.CIFAR10(root="./data", train=True, download=True)
        adapter = TorchVisionClassificationAdapter(ds)
        # adapter[i] → {"inputs": PIL.Image, "gt_label": int}
    """

    def __init__(self, dataset: Any) -> None:
        self._ds = dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img, label = self._ds[idx]
        return {"inputs": img, "gt_label": int(label)}


class TorchVisionDetectionAdapter:
    """Wrap a torchvision detection dataset (VOC, CocoDetection) for detection.

    Usage::

        import torchvision.datasets as tvds
        ds = tvds.VOCDetection(root="./data", year="2012", image_set="train", download=True)
        adapter = TorchVisionDetectionAdapter(ds)
        # adapter[i] → {"inputs": PIL.Image, "gt_bboxes": np.ndarray, "gt_labels": np.ndarray}
    """

    def __init__(self, dataset: Any) -> None:
        self._ds = dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img, target = self._ds[idx]
        # target varies by dataset; attempt to normalise
        if isinstance(target, dict):
            boxes = _extract_boxes(target)
            labels = _extract_labels(target)
        elif isinstance(target, (list, tuple)):
            boxes, labels = target[0], target[1]
            boxes = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        return {
            "inputs": img,
            "gt_bboxes": boxes,
            "gt_labels": labels,
        }


def _extract_boxes(target: dict) -> np.ndarray:
    """Extract boxes from a detection target dict (handles VOC, COCO, etc.)."""
    for key in ("boxes", "bbox", "bboxes", "annotation"):
        if key in target:
            val = target[key]
            if isinstance(val, dict) and "bbox" in val:
                # VOC format: {"annotation": {"object": [{"bndbox": ...}]}}
                objects = val.get("object", [])
                if not isinstance(objects, list):
                    objects = [objects]
                boxes = []
                for obj in objects:
                    b = obj.get("bndbox", {})
                    boxes.append([
                        float(b.get("xmin", 0)), float(b.get("ymin", 0)),
                        float(b.get("xmax", 0)), float(b.get("ymax", 0)),
                    ])
                return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            return np.asarray(val, dtype=np.float32).reshape(-1, 4)
    return np.zeros((0, 4), dtype=np.float32)


def _extract_labels(target: dict) -> np.ndarray:
    """Extract labels from a detection target dict."""
    for key in ("labels", "category_id", "name"):
        if key in target:
            val = target[key]
            if isinstance(val, (int, float)):
                return np.array([int(val)], dtype=np.int64)
            return np.asarray(val, dtype=np.int64).flatten()
    # VOC: labels embedded in annotation objects
    ann = target.get("annotation", {})
    objects = ann.get("object", [])
    if not isinstance(objects, list):
        objects = [objects]
    labels = [int(obj.get("name", 0)) for obj in objects]
    return np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)