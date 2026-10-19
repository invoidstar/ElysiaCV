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
    """Load and format annotations for detection / segmentation tasks.

    Pre-populated fields (by dataset) are formatted into the standard
    ElysiaCV convention: numpy arrays in xyxy float32 format.

    Args:
        with_bbox: Load bounding boxes.
        with_mask: Load instance masks.
        with_seg: Load semantic segmentation mask.
    """

    def __init__(self, with_bbox: bool = True, with_mask: bool = False, with_seg: bool = False) -> None:
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_seg = with_seg

    def __call__(self, results: dict) -> dict:
        # Bounding boxes: ensure numpy float32 in xyxy format
        if self.with_bbox and "gt_bboxes" in results:
            bboxes = results["gt_bboxes"]
            if isinstance(bboxes, list) and len(bboxes) > 0:
                if isinstance(bboxes[0], (list, np.ndarray)):
                    results["gt_bboxes"] = np.array(bboxes, dtype=np.float32)
                else:
                    results["gt_bboxes"] = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
            elif isinstance(bboxes, list) and len(bboxes) == 0:
                results["gt_bboxes"] = np.zeros((0, 4), dtype=np.float32)
            elif isinstance(bboxes, np.ndarray):
                results["gt_bboxes"] = bboxes.astype(np.float32)

        # Labels: ensure numpy int64
        if self.with_bbox and "gt_labels" in results:
            labels = results["gt_labels"]
            if isinstance(labels, list):
                results["gt_labels"] = np.array(labels, dtype=np.int64)
            elif isinstance(labels, np.ndarray):
                results["gt_labels"] = labels.astype(np.int64)

        # Semantic segmentation mask
        if self.with_seg and "gt_semantic_seg" in results:
            seg = results["gt_semantic_seg"]
            if isinstance(seg, np.ndarray):
                results["gt_semantic_seg"] = seg.astype(np.int64)

        # Store metadata
        if "gt_bboxes" in results:
            results["gt_bboxes_ignore"] = results.get(
                "gt_bboxes_ignore", np.zeros((0, 4), dtype=np.float32))
            results["gt_labels_ignore"] = results.get(
                "gt_labels_ignore", np.zeros((0,), dtype=np.int64))

        return results