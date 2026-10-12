"""Output formatting transforms — convert to tensor, pack batch-ready data."""

from __future__ import annotations

import numpy as np

from elysia_cv.core.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackInputs:
    """Pack image into batch-ready dict.

    Converts numpy array to torch tensor with C×H×W layout.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, results: dict) -> dict:
        import torch

        img = results["img"]
        if isinstance(img, np.ndarray):
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)  # HWC → CHW
            img = torch.from_numpy(np.ascontiguousarray(img))
        results["inputs"] = img.float() if img.dtype in (np.float32, np.float64) else img

        # Clean up intermediate fields
        results.pop("img", None)
        return results


@TRANSFORMS.register_module()
class PackDetInputs(PackInputs):
    """Pack detection-specific data (images + ground-truth bboxes/labels)."""

    def __call__(self, results: dict) -> dict:
        results = super().__call__(results)
        import torch
        if "gt_bboxes" in results:
            results["gt_bboxes"] = torch.tensor(np.array(results["gt_bboxes"], dtype=np.float32))
        if "gt_labels" in results:
            results["gt_labels"] = torch.tensor(np.array(results["gt_labels"], dtype=np.int64))
        return results


@TRANSFORMS.register_module()
class PackSegInputs(PackInputs):
    """Pack segmentation-specific data (images + ground-truth masks)."""

    def __call__(self, results: dict) -> dict:
        results = super().__call__(results)
        import torch
        if "gt_semantic_seg" in results:
            mask = results["gt_semantic_seg"]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
            results["gt_semantic_seg"] = mask
        return results