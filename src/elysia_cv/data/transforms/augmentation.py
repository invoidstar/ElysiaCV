"""Data augmentation transforms."""

from __future__ import annotations

import numpy as np

from elysia_cv.core.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomFlip:
    """Random horizontal flip.

    Args:
        prob: Flip probability.
        direction: ``"horizontal"`` or ``"vertical"``.
    """

    def __init__(self, prob: float = 0.5, direction: str = "horizontal") -> None:
        self.prob = prob
        self.direction = direction

    def __call__(self, results: dict) -> dict:
        import random
        if random.random() < self.prob:
            img = results["img"]
            if self.direction == "horizontal":
                img = np.fliplr(img)
            elif self.direction == "vertical":
                img = np.flipud(img)
            results["img"] = img
            results["flip"] = True
            results["flip_direction"] = self.direction

            # Also flip bboxes if present
            if "gt_bboxes" in results and len(results["gt_bboxes"]) > 0:
                w = img.shape[1]
                for bbox in results["gt_bboxes"]:
                    x1, y1, x2, y2 = bbox
                    if self.direction == "horizontal":
                        bbox[:] = [w - x2, y1, w - x1, y2]
        else:
            results["flip"] = False
        return results


@TRANSFORMS.register_module()
class ColorJitter:
    """Random color jitter.

    Args:
        brightness: Max brightness delta.
        contrast: Max contrast delta.
        saturation: Max saturation delta.
        hue: Max hue delta (not applicable to RGB numpy; applied via PIL).
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, results: dict) -> dict:
        from PIL import Image, ImageEnhance
        import random

        img = results["img"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        if self.brightness > 0:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast > 0:
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if self.saturation > 0:
            factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)

        results["img"] = np.array(img)
        return results


@TRANSFORMS.register_module()
class RandomErasing:
    """Random erasing augmentation.

    Args:
        prob: Erasing probability.
        scale: Range of erased area proportion.
        ratio: Range of aspect ratio.
    """

    def __init__(self, prob: float = 0.5, scale: tuple[float, float] = (0.02, 0.33),
                 ratio: tuple[float, float] = (0.3, 3.3)) -> None:
        self.prob = prob
        self.scale = scale
        self.ratio = ratio

    def __call__(self, results: dict) -> dict:
        import random
        if random.random() > self.prob:
            return results
        img = results["img"]
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)
            erase_h = int(np.sqrt(target_area * aspect))
            erase_w = int(np.sqrt(target_area / aspect))
            if erase_h < h and erase_w < w:
                x1 = random.randint(0, h - erase_h)
                y1 = random.randint(0, w - erase_w)
                img[x1:x1 + erase_h, y1:y1 + erase_w] = np.random.randint(0, 256, (erase_h, erase_w, img.shape[2]))
                break
        results["img"] = img
        return results


@TRANSFORMS.register_module()
class PhotoMetricDistortion:
    """Photometric distortion for segmentation — brightness, contrast, saturation, hue."""

    def __init__(self, brightness_delta: int = 32, contrast_range: tuple[float, float] = (0.5, 1.5),
                 saturation_range: tuple[float, float] = (0.5, 1.5), hue_delta: int = 18) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results: dict) -> dict:
        from PIL import Image, ImageEnhance
        import random
        img = results["img"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        # Brightness
        if random.randint(0, 1):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = ImageEnhance.Brightness(img).enhance(1 + delta / 255.0)
        # Contrast
        contrast = random.uniform(*self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        # Saturation
        saturation = random.uniform(*self.saturation_range)
        img = ImageEnhance.Color(img).enhance(saturation)
        # Hue
        if random.randint(0, 1):
            hue = random.uniform(-self.hue_delta, self.hue_delta)
            img_np = np.array(img.convert("HSV"))
            img_np[:, :, 0] = (img_np[:, :, 0].astype(int) + int(hue)) % 180
            img = Image.fromarray(img_np, "HSV").convert("RGB")

        results["img"] = np.array(img)
        return results