"""Image loading transforms."""

from __future__ import annotations

from PIL import Image
import numpy as np

from elysia_cv.core.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile:
    """Load an image from file path.

    Expects ``results["img_path"]``; sets ``results["img"]`` as numpy array (H, W, C) RGB.
    """

    def __init__(self, to_float32: bool = False, backend: str = "pillow") -> None:
        self.to_float32 = to_float32
        self.backend = backend

    def __call__(self, results: dict) -> dict:
        path = results["img_path"]
        img = Image.open(path).convert("RGB")
        img = np.array(img, dtype=np.float32 if self.to_float32 else np.uint8)
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


@TRANSFORMS.register_module()
class LoadAnnotations:
    """Load annotations — placeholder for detection/segmentation tasks.

    Expects ``results["ann_path"]``; loads JSON or other format.
    """

    def __init__(self, with_bbox: bool = True, with_mask: bool = False, with_seg: bool = False) -> None:
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_seg = with_seg

    def __call__(self, results: dict) -> dict:
        # In real implementation, parse annotation file
        results["gt_bboxes"] = results.get("gt_bboxes", [])
        results["gt_labels"] = results.get("gt_labels", [])
        return results