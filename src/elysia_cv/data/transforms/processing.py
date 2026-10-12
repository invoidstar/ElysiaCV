"""Image processing transforms — resize, crop, normalize."""

from __future__ import annotations

import numpy as np

from elysia_cv.core.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Resize:
    """Resize image to a target size.

    Args:
        size: (width, height) or single int for short edge.
        keep_ratio: If True, resize keeping aspect ratio.
    """

    def __init__(self, size: int | tuple[int, int] = 224, keep_ratio: bool = False) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, results: dict) -> dict:
        from PIL import Image
        img = results["img"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if self.keep_ratio:
            img.thumbnail(self.size, Image.Resampling.LANCZOS)
        else:
            img = img.resize(self.size, Image.Resampling.LANCZOS)
        results["img"] = np.array(img)
        results["img_shape"] = results["img"].shape
        return results


@TRANSFORMS.register_module()
class RandomResizedCrop:
    """Random resize and crop augmentation.

    Args:
        size: Output size (H, W).
        scale: Range of area fraction kept before cropping.
    """

    def __init__(self, size: int | tuple[int, int] = 224, scale: tuple[float, float] = (0.08, 1.0)) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale

    def __call__(self, results: dict) -> dict:
        from PIL import Image
        import random

        img = results["img"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        w, h = img.size
        area = w * h
        target_area = random.uniform(*self.scale) * area
        aspect_ratio = random.uniform(3.0 / 4.0, 4.0 / 3.0)
        crop_w = int(np.sqrt(target_area * aspect_ratio))
        crop_h = int(np.sqrt(target_area / aspect_ratio))
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        img = img.resize(self.size, Image.Resampling.LANCZOS)

        results["img"] = np.array(img)
        results["img_shape"] = results["img"].shape
        return results


@TRANSFORMS.register_module()
class CenterCrop:
    """Center crop to target size."""

    def __init__(self, size: int | tuple[int, int] = 224) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, results: dict) -> dict:
        from PIL import Image
        img = results["img"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        w, h = img.size
        crop_w, crop_h = self.size
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))

        results["img"] = np.array(img)
        results["img_shape"] = results["img"].shape
        return results


@TRANSFORMS.register_module()
class Normalize:
    """Normalize image with mean and std.

    Args:
        mean: Per-channel mean (R, G, B).
        std: Per-channel std.
    """

    def __init__(self, mean: list[float] | None = None, std: list[float] | None = None) -> None:
        self.mean = np.array(mean or [123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array(std or [58.395, 57.12, 57.375], dtype=np.float32)

    def __call__(self, results: dict) -> dict:
        img = results["img"].astype(np.float32)
        img = (img - self.mean) / self.std
        results["img"] = img
        results["img_norm_cfg"] = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        return results