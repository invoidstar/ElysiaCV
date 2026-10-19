"""Data augmentation transforms."""

from __future__ import annotations

import numpy as np
import torch

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
                bboxes = results["gt_bboxes"]
                if self.direction == "horizontal":
                    x1 = bboxes[:, 0].copy()
                    x2 = bboxes[:, 2].copy()
                    bboxes[:, 0] = w - x2
                    bboxes[:, 2] = w - x1
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


@TRANSFORMS.register_module()
class RandomRotation:
    """Random rotation with bbox coordinate update.

    Rotates the image by a random angle and adjusts bounding-box coordinates
    accordingly.  The image canvas is expanded to fit the rotated content
    (i.e. no cropping).

    Args:
        degrees: Maximum rotation angle (±degrees).
        prob: Probability of applying the rotation.
        expand: If True, expand canvas to fit rotated image.  Default True.
    """

    def __init__(self, degrees: float = 10.0, prob: float = 0.5, expand: bool = True) -> None:
        self.degrees = degrees
        self.prob = prob
        self.expand = expand

    def __call__(self, results: dict) -> dict:
        import random
        from PIL import Image as PILImage

        if random.random() > self.prob:
            return results

        angle = random.uniform(-self.degrees, self.degrees)
        img = results["img"]
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            pil_img = PILImage.fromarray(img.astype(np.uint8))
        else:
            h, w = img.size[::-1]
            pil_img = img

        # Rotate image
        if self.expand:
            rotated = pil_img.rotate(angle, resample=PILImage.BILINEAR, expand=True)
        else:
            rotated = pil_img.rotate(angle, resample=PILImage.BILINEAR, expand=False)

        results["img"] = np.array(rotated)

        # Transform bboxes
        if "gt_bboxes" in results and len(results["gt_bboxes"]) > 0:
            rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            cx, cy = w / 2.0, h / 2.0

            bboxes = results["gt_bboxes"].copy()  # [N, 4] x1,y1,x2,y2
            corners = np.array([
                bboxes[:, [0, 1]],  # x1,y1
                bboxes[:, [2, 1]],  # x2,y1
                bboxes[:, [2, 3]],  # x2,y2
                bboxes[:, [0, 3]],  # x1,y2
            ])  # [4, N, 2]
            corners = corners.transpose(1, 0, 2).reshape(-1, 2)  # [4N, 2]

            # Rotate around center
            tx = corners[:, 0] - cx
            ty = corners[:, 1] - cy
            rx = tx * cos_a - ty * sin_a + cx
            ry = tx * sin_a + ty * cos_a + cy

            corners_rot = np.stack([rx, ry], axis=1).reshape(-1, 4, 2)  # [N, 4, 2]
            new_x1 = corners_rot[:, :, 0].min(axis=1)
            new_y1 = corners_rot[:, :, 1].min(axis=1)
            new_x2 = corners_rot[:, :, 0].max(axis=1)
            new_y2 = corners_rot[:, :, 1].max(axis=1)

            # Adjust for canvas expansion
            if self.expand:
                new_h, new_w = rotated.height, rotated.width
                dx = (new_w - w) / 2.0
                dy = (new_h - h) / 2.0
                new_x1 += dx
                new_y1 += dy
                new_x2 += dx
                new_y2 += dy

            results["gt_bboxes"] = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
            results["gt_bboxes"] = np.clip(results["gt_bboxes"], 0, None)

        return results

    def __repr__(self) -> str:
        return f"RandomRotation(degrees={self.degrees}, prob={self.prob})"


@TRANSFORMS.register_module()
class RandomAffine:
    """Random affine transformation with bbox coordinate update.

    Applies random translation, scaling, and shearing.  Optionally the
    image canvas is expanded to avoid clipping.

    Args:
        translate: Max translation ratio [tx, ty] relative to image size.
        scale: Scale range [min, max].
        shear: Max shear angle in degrees.
        prob: Probability of applying the transform.
    """

    def __init__(
        self,
        translate: tuple[float, float] = (0.1, 0.1),
        scale: tuple[float, float] = (0.9, 1.1),
        shear: float = 5.0,
        prob: float = 0.5,
    ) -> None:
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.prob = prob

    def __call__(self, results: dict) -> dict:
        import random
        from PIL import Image as PILImage

        if random.random() > self.prob:
            return results

        img = results["img"]
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            pil_img = PILImage.fromarray(img.astype(np.uint8))
        else:
            h, w = img.size[::-1]
            pil_img = img

        # Random parameters
        tx = random.uniform(-self.translate[0], self.translate[0]) * w
        ty = random.uniform(-self.translate[1], self.translate[1]) * h
        s = random.uniform(*self.scale)
        shear_angle = random.uniform(-self.shear, self.shear)

        # Build PIL affine matrix: [a, b, c, d, e, f]
        # Transform: x' = a*x + b*y + c, y' = d*x + e*y + f
        shear_rad = np.deg2rad(shear_angle)
        a = s * np.cos(shear_rad)
        b = -s * np.sin(shear_rad)
        d = s * np.sin(shear_rad)
        e = s * np.cos(shear_rad)
        c = tx
        f = ty

        # PIL affine transform (inverse matrix convention: maps output→input)
        # We need the inverse: given x',y' find x,y
        det = a * e - b * d
        if abs(det) < 1e-8:
            return results
        inv_a = e / det
        inv_b = -b / det
        inv_c = (b * f - e * c) / det
        inv_d = -d / det
        inv_e = a / det
        inv_f = (d * c - a * f) / det

        new_w = int(w * s + abs(tx))
        new_h = int(h * s + abs(ty))
        new_w = max(new_w, 1)
        new_h = max(new_h, 1)

        affine_img = pil_img.transform(
            (new_w, new_h),
            PILImage.AFFINE,
            (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f),
            resample=PILImage.BILINEAR,
        )
        results["img"] = np.array(affine_img)

        # Transform bboxes
        if "gt_bboxes" in results and len(results["gt_bboxes"]) > 0:
            bboxes = results["gt_bboxes"].copy()  # [N, 4] x1,y1,x2,y2
            corners = np.array([
                bboxes[:, [0, 1]], bboxes[:, [2, 1]],
                bboxes[:, [2, 3]], bboxes[:, [0, 3]],
            ]).transpose(1, 0, 2).reshape(-1, 2)  # [4N, 2]

            x = corners[:, 0]
            y = corners[:, 1]
            tx_corners = a * x + b * y + c
            ty_corners = d * x + e * y + f

            corners_aff = np.stack([tx_corners, ty_corners], axis=1).reshape(-1, 4, 2)
            new_x1 = corners_aff[:, :, 0].min(axis=1)
            new_y1 = corners_aff[:, :, 1].min(axis=1)
            new_x2 = corners_aff[:, :, 0].max(axis=1)
            new_y2 = corners_aff[:, :, 1].max(axis=1)

            results["gt_bboxes"] = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
            results["gt_bboxes"] = np.clip(results["gt_bboxes"], 0, None)

        return results

    def __repr__(self) -> str:
        return f"RandomAffine(scale={self.scale}, prob={self.prob})"


# ---------------------------------------------------------------------------
# Batch-level augmentations (operate on collated batch dicts)
# ---------------------------------------------------------------------------


@TRANSFORMS.register_module()
class MixUp:
    """MixUp batch augmentation for image classification.

    For each batch, images are linearly interpolated with a randomly shuffled
    copy of the same batch::

        mixed_img  = lam · img + (1 - lam) · img_shuffled
        soft_label = lam · onehot(label) + (1 - lam) · onehot(label_shuffled)

    where ``lam ~ Beta(alpha, alpha)``.  The soft label is stored as
    ``gt_label_soft`` in the batch dict; the original hard label is preserved
    as ``gt_label_orig`` for metrics.

    Args:
        alpha: Beta distribution parameter (default 0.2, mild mixing).
            Larger α → stronger mixing.  0.0 disables MixUp.
        prob: Probability of applying MixUp to a batch.
        num_classes: Total number of classes (required for one-hot encoding).
    """

    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 1.0,
        num_classes: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, batch: dict) -> dict:
        """Apply MixUp to a collated batch dict.

        Expected batch keys: ``inputs`` [B,C,H,W], ``gt_label`` [B].
        Adds: ``gt_label_soft`` [B,C], ``gt_label_orig`` [B], ``mix_lam``.
        """
        import random

        if self.alpha <= 0 or random.random() > self.prob:
            return batch

        inputs = batch["inputs"]  # tensor [B, C, H, W]
        labels = batch.get("gt_label")
        if labels is None or not isinstance(inputs, torch.Tensor):
            return batch

        B = inputs.size(0)
        lam = float(np.random.beta(self.alpha, self.alpha))
        lam = max(lam, 1.0 - lam)  # clamp to ≥0.5 for stability
        index = torch.randperm(B, device=inputs.device)

        # Mix images
        batch["inputs"] = lam * inputs + (1.0 - lam) * inputs[index]

        # Mix labels → soft one-hot
        if self.num_classes is None:
            self.num_classes = int(labels.max().item()) + 1

        a_onehot = torch.zeros(B, self.num_classes, device=inputs.device)
        a_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        b_onehot = torch.zeros(B, self.num_classes, device=inputs.device)
        b_onehot.scatter_(1, labels[index].unsqueeze(1), 1.0)

        batch["gt_label_orig"] = labels
        batch["gt_label_soft"] = lam * a_onehot + (1.0 - lam) * b_onehot
        batch["mix_lam"] = lam

        return batch

    def __repr__(self) -> str:
        return f"MixUp(alpha={self.alpha}, prob={self.prob})"


@TRANSFORMS.register_module()
class CutMix:
    """CutMix batch augmentation for image classification.

    Replaces a rectangular region of each image with a patch from another
    image in the batch::

        mask = binary_mask(bbox)
        mixed_img = mask * img + (1 - mask) * img_shuffled
        soft_label = lam * onehot(label) + (1 - lam) * onehot(label_shuffled)

    where ``lam`` is the area ratio of the kept region.  The soft label is
    stored as ``gt_label_soft``; original labels as ``gt_label_orig``.

    Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features", ICCV 2019.

    Args:
        alpha: Beta distribution parameter for the cut-region area ratio.
            Larger α → more balanced cuts.  0.0 disables CutMix.
        prob: Probability of applying CutMix to a batch.
        num_classes: Total number of classes (required for one-hot encoding).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 1.0,
        num_classes: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, batch: dict) -> dict:
        """Apply CutMix to a collated batch dict.

        Expected batch keys: ``inputs`` [B,C,H,W], ``gt_label`` [B].
        Adds: ``gt_label_soft`` [B,C], ``gt_label_orig`` [B], ``mix_lam``.
        """
        import random

        if self.alpha <= 0 or random.random() > self.prob:
            return batch

        inputs = batch["inputs"]  # tensor [B, C, H, W]
        labels = batch.get("gt_label")
        if labels is None or not isinstance(inputs, torch.Tensor):
            return batch

        B, C, H, W = inputs.shape
        lam = float(np.random.beta(self.alpha, self.alpha))
        index = torch.randperm(B, device=inputs.device)

        # Compute cut region size from lam (lam = kept area ratio)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Random top-left corner
        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        # Build binary mask: 1 = keep, 0 = replace
        mask = torch.ones(B, C, H, W, device=inputs.device)
        mask[:, :, y1:y2, x1:x2] = 0.0

        # CutMix: keep region from original, fill cut region from shuffled
        batch["inputs"] = mask * inputs + (1.0 - mask) * inputs[index]

        # Update lam to actual area ratio
        actual_lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)

        # Mix labels → soft one-hot
        if self.num_classes is None:
            self.num_classes = int(labels.max().item()) + 1

        a_onehot = torch.zeros(B, self.num_classes, device=inputs.device)
        a_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        b_onehot = torch.zeros(B, self.num_classes, device=inputs.device)
        b_onehot.scatter_(1, labels[index].unsqueeze(1), 1.0)

        batch["gt_label_orig"] = labels
        batch["gt_label_soft"] = actual_lam * a_onehot + (1.0 - actual_lam) * b_onehot
        batch["mix_lam"] = actual_lam

        return batch

    def __repr__(self) -> str:
        return f"CutMix(alpha={self.alpha}, prob={self.prob})"


@TRANSFORMS.register_module()
class Mosaic:
    """Mosaic augmentation for object detection (YOLO-style).

    Stitches 4 images into a 2×2 grid.  Each group of 4 images in the
    batch produces one mosaic output.  Bounding-box coordinates are
    updated to reflect the new positions.

    This is a **batch-level** transform.  The batch size should be
    divisible by 4 for full coverage.

    Args:
        output_size: Target (height, width) for the mosaic canvas.
        prob: Probability of applying Mosaic to each group of 4.
        center_jitter: Random offset range for the quadrant center
            (fraction of output_size).  Default 0.2 = ±20% jitter.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = (640, 640),
        prob: float = 1.0,
        center_jitter: float = 0.2,
    ) -> None:
        self.output_size = output_size
        self.prob = prob
        self.center_jitter = center_jitter

    def __call__(self, batch: dict) -> dict:
        """Apply Mosaic to each group of 4 images in the batch.

        Expected batch keys:
          - ``inputs`` [B, C, H, W] tensor
          - ``gt_bboxes`` [B, N, 4]  or  list of tensors
          - ``gt_labels``  [B, N]     or  list of tensors

        Sizes the mosaic canvas to ``output_size``.  Each source image is
        resized to fill its quadrant before pasting.
        """
        import random

        if random.random() > self.prob:
            return batch

        inputs = batch.get("inputs")
        if inputs is None or not isinstance(inputs, torch.Tensor):
            return batch

        B, C, orig_H, orig_W = inputs.shape
        if B < 4:
            return batch

        out_H, out_W = self.output_size
        device = inputs.device

        # Group into 4s
        n_mosaics = B // 4
        mosaic_inputs = torch.zeros(n_mosaics, C, out_H, out_W, device=device, dtype=inputs.dtype)

        # Handle bboxes/labels — may be tensor [B, N, 4] or list of tensors
        has_bboxes = "gt_bboxes" in batch and batch["gt_bboxes"] is not None
        has_labels = "gt_labels" in batch and batch["gt_labels"] is not None

        # Collect results
        result_bboxes: list[torch.Tensor] = []
        result_labels: list[torch.Tensor] = []

        for m in range(n_mosaics):
            # Random center (with jitter)
            cx = int(out_W / 2 + random.uniform(-self.center_jitter, self.center_jitter) * out_W)
            cy = int(out_H / 2 + random.uniform(-self.center_jitter, self.center_jitter) * out_H)
            cx = max(0, min(cx, out_W))
            cy = max(0, min(cy, out_H))

            canvas = torch.zeros(out_H, out_W, 3) if C == 3 else torch.zeros(out_H, out_W, C)  # temporary HWC

            all_bboxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            # 4 quadrants: top-left, top-right, bottom-left, bottom-right
            placements = [
                (0, 0, cx, cy),               # TL
                (cx, 0, out_W, cy),            # TR
                (0, cy, cx, out_H),            # BL
                (cx, cy, out_W, out_H),        # BR
            ]

            for k, (px1, py1, px2, py2) in enumerate(placements):
                img_idx = m * 4 + k
                img = inputs[img_idx]  # [C, H, W]
                pw, ph = px2 - px1, py2 - py1

                # Resize image to fill quadrant
                img_np = img.permute(1, 2, 0).cpu().numpy()  # HWC
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(
                    (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0
                    else img_np.astype(np.uint8)
                )
                pil_img = pil_img.resize((pw, ph), PILImage.BILINEAR)
                pasted = np.array(pil_img).astype(np.float32)
                if pasted.max() > 1.0:
                    pasted = pasted / 255.0

                # Tensor back to canvas
                canvas_np = canvas.numpy() if not isinstance(canvas, np.ndarray) else canvas
                # Paste into canvas
                if C == 3:
                    canvas[py1:py2, px1:px2] = torch.from_numpy(pasted)

                # Transform bboxes from original → mosaic coords
                if has_bboxes:
                    bbs = batch["gt_bboxes"]
                    if isinstance(bbs, torch.Tensor):
                        bbs = bbs[img_idx]  # [N, 4]
                        if bbs.numel() == 0:
                            continue
                        # Scale from original size to quadrant size
                        scale_x = pw / orig_W
                        scale_y = ph / orig_H
                        bb = bbs.clone().to(torch.float32)
                        bb[:, [0, 2]] = bb[:, [0, 2]] * scale_x + px1
                        bb[:, [1, 3]] = bb[:, [1, 3]] * scale_y + py1
                        all_bboxes.append(bb)
                    elif isinstance(bbs, list):
                        bb = bbs[img_idx].clone().to(torch.float32)
                        if bb.numel() == 0:
                            continue
                        scale_x = pw / orig_W
                        scale_y = ph / orig_H
                        bb[:, [0, 2]] = bb[:, [0, 2]] * scale_x + px1
                        bb[:, [1, 3]] = bb[:, [1, 3]] * scale_y + py1
                        all_bboxes.append(bb)

                if has_labels:
                    lbs = batch["gt_labels"]
                    lb = lbs[img_idx] if isinstance(lbs, torch.Tensor) else lbs[img_idx]
                    if lb.numel() > 0:
                        all_labels.append(lb)

            # Assemble mosaic
            mosaic_inputs[m] = canvas.permute(2, 0, 1)  # HWC → CHW

            if all_bboxes:
                result_bboxes.append(torch.cat(all_bboxes, dim=0))
            else:
                result_bboxes.append(torch.zeros(0, 4, device=device))
            if all_labels:
                result_labels.append(torch.cat(all_labels, dim=0))
            else:
                result_labels.append(torch.zeros(0, dtype=torch.long, device=device))

        # Replace batch
        batch["inputs"] = mosaic_inputs
        if has_bboxes:
            # Keep as list of tensors (variable N per image)
            batch["gt_bboxes"] = result_bboxes
        if has_labels:
            batch["gt_labels"] = result_labels

        return batch

    def __repr__(self) -> str:
        return f"Mosaic(output_size={self.output_size}, prob={self.prob})"