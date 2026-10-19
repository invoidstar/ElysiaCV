"""Test-Time Augmentation (TTA) — inference-time multi-scale + flip voting.

For classification, predictions from different augmentations are averaged
(logits or probabilities).  For detection / segmentation, TTA is more
involved (bbox merging, mask averaging) and is handled separately.

Usage::

    from elysia_cv.data.transforms.tta import tta_predict

    preds = tta_predict(model, image, device,
                        scales=[0.8, 1.0, 1.2],
                        flips=[False, True])
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def tta_predict(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    *,
    scales: list[float] | None = None,
    flips: list[bool] | None = None,
    transform_fn: Callable | None = None,
) -> torch.Tensor:
    """Run TTA for a single image and return averaged logits.

    The image is expected to be a tensor of shape ``[C, H, W]`` already
    preprocessed (resized, normalised).  The function applies the requested
    scales and horizontal flips, runs the model on each variant, and returns
    the mean of the output logits.

    Args:
        model: Classification model (``model(input_image)`` → logits tensor).
        image: Preprocessed image tensor ``[C, H, W]`` (already on CPU or GPU).
        device: CUDA / CPU device.
        scales: List of scale factors relative to 1.0.  Default: ``[1.0]``.
        flips: List of flip flags.  ``True`` = horizontal mirror.  Default: ``[False]``.
        transform_fn: Optional per-variant callback ``fn(image, scale, flip)``
            that returns the transformed tensor.  If None, uses built-in
            ``_default_tta_transform``.

    Returns:
        Averaged logits ``[1, num_classes]`` on *device*.

    Example::

        # Basic TTA with 3 scales and flips (6 predictions averaged)
        logits = tta_predict(model, image, device,
                             scales=[0.8, 1.0, 1.2],
                             flips=[False, True])
        pred_class = logits.argmax(dim=1).item()
    """
    if scales is None:
        scales = [1.0]
    if flips is None:
        flips = [False]

    tf = transform_fn if transform_fn is not None else _default_tta_transform
    model.eval()

    all_logits: list[torch.Tensor] = []

    with torch.no_grad():
        for scale in scales:
            for flip in flips:
                variant = tf(image.clone(), scale, flip)
                variant = variant.unsqueeze(0).to(device)  # [1, C, H, W]
                output = model(variant)
                if isinstance(output, dict):
                    logits = output.get("logits", output)
                else:
                    logits = output
                all_logits.append(logits)  # [1, num_classes]

    # Stack and average
    stacked = torch.cat(all_logits, dim=0)  # [N_variants, num_classes]
    return stacked.mean(dim=0, keepdim=True)  # [1, num_classes]


def _default_tta_transform(
    image: torch.Tensor, scale: float, flip: bool
) -> torch.Tensor:
    """Default per-variant transform: resize by *scale*, center-crop/pad to
    the original spatial size, optionally horizontal-flip.

    Assumes *image* is ``[C, H, W]``.
    """
    _, orig_h, orig_w = image.shape
    new_h, new_w = max(1, int(orig_h * scale)), max(1, int(orig_w * scale))

    if scale != 1.0:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # Center-crop or pad back to original spatial size
    image = _center_crop_or_pad(image, orig_h, orig_w)

    if flip:
        image = torch.flip(image, dims=[-1])  # horizontal mirror

    return image


def _center_crop_or_pad(
    image: torch.Tensor, target_h: int, target_w: int
) -> torch.Tensor:
    """Center-crop if larger, zero-pad if smaller, to match (target_h, target_w)."""
    _, h, w = image.shape

    # Vertical
    if h > target_h:
        start_h = (h - target_h) // 2
        image = image[:, start_h:start_h + target_h, :]
    elif h < target_h:
        pad_h = target_h - h
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        image = F.pad(image, (0, 0, pad_top, pad_bot))

    # Horizontal
    _, h, w = image.shape
    if w > target_w:
        start_w = (w - target_w) // 2
        image = image[:, :, start_w:start_w + target_w]
    elif w < target_w:
        pad_w = target_w - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = F.pad(image, (pad_left, pad_right))

    return image


class TTAClassificationWrapper:
    """Model wrapper that applies TTA automatically.

    Usage::

        model = TTAClassificationWrapper(model,
                                         scales=[0.8, 1.0, 1.2],
                                         flips=[False, True])
        logits = model(image)  # [B, num_classes] — internally runs TTA per image
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        scales: list[float] | None = None,
        flips: list[bool] | None = None,
    ) -> None:
        self.model = model
        self.scales = scales or [1.0]
        self.flips = flips or [False]
        self.device = next(model.parameters()).device

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Run TTA for each image in the batch.

        ``images`` is ``[B, C, H, W]`` on the appropriate device.
        Returns ``[B, num_classes]`` logits.
        """
        B = images.size(0)
        results: list[torch.Tensor] = []
        for i in range(B):
            single = tta_predict(
                self.model,
                images[i],
                self.device,
                scales=self.scales,
                flips=self.flips,
            )
            results.append(single)
        return torch.cat(results, dim=0)  # [B, num_classes]