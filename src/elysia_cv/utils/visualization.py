"""Detection & segmentation visualization utilities.

Draw bounding boxes, instance masks, and semantic segmentation overlays
onto images for debugging and qualitative analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Colour palette (BGR order for OpenCV compat, stored as RGB for PIL) ──

_COLORS_RGB = [
    (0xDC, 0x14, 0x3C),  # crimson
    (0x00, 0xBF, 0xFF),  # deep sky blue
    (0x32, 0xCD, 0x32),  # lime green
    (0xFF, 0xD7, 0x00),  # gold
    (0xFF, 0x69, 0xB4),  # hot pink
    (0x8A, 0x2B, 0xE2),  # blue violet
    (0x00, 0xCE, 0xD1),  # dark turquoise
    (0xFF, 0x7F, 0x50),  # coral
    (0xAD, 0xD8, 0xE6),  # light steel blue
    (0xFF, 0xA0, 0x7A),  # light salmon
]


def _get_color(idx: int) -> tuple[int, int, int]:
    return _COLORS_RGB[idx % len(_COLORS_RGB)]


# ── Detection (bounding boxes) ──


def draw_bboxes(
    image: Image.Image | np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    *,
    class_names: list[str] | None = None,
    score_thr: float = 0.0,
    line_width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """Draw bounding boxes onto an image.

    Args:
        image: PIL Image or H×W×3 numpy array (RGB).
        bboxes: N×4 array of ``[x1, y1, x2, y2]`` (absolute coords).
        labels: length-N int array of category IDs.
        scores: length-N float array of confidence scores.
        class_names: optional list mapping label ID → name string.
        score_thr: only draw boxes with score ≥ this threshold.
        line_width: box border width in pixels.
        font_size: label text size.

    Returns:
        PIL Image with boxes drawn.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image or numpy array, got {type(image)}")

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i in range(len(bboxes)):
        if scores is not None and scores[i] < score_thr:
            continue

        x1, y1, x2, y2 = bboxes[i].tolist()
        label_id = int(labels[i]) if labels is not None else 0
        color = _get_color(label_id)

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Build label text
        parts = []
        if class_names and label_id < len(class_names):
            parts.append(class_names[label_id])
        else:
            parts.append(f"cls{label_id}")
        if scores is not None:
            parts.append(f"{scores[i]:.2f}")

        text = " ".join(parts)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Background rectangle for label
        draw.rectangle(
            [x1, max(0, y1 - text_h - 2), x1 + text_w + 2, y1],
            fill=color,
        )
        # Text in white or black depending on background brightness
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        draw.text((x1 + 1, max(0, y1 - text_h - 1)), text, fill=text_color, font=font)

    return image


# ── Segmentation overlay ──


def draw_segmentation_overlay(
    image: Image.Image | np.ndarray,
    seg_map: np.ndarray,
    *,
    alpha: float = 0.5,
    class_names: list[str] | None = None,
) -> Image.Image:
    """Overlay a semantic segmentation map onto an image.

    Args:
        image: PIL Image or H×W×3 numpy array (RGB).
        seg_map: H×W int array of class IDs.
        alpha: blend factor (0 = pure image, 1 = pure overlay).
        class_names: optional list of class name strings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.astype(np.uint8)

    # Build colour overlay
    overlay = np.zeros_like(img_np)
    unique_ids = np.unique(seg_map)
    cmap = plt.cm.get_cmap("tab20", max(unique_ids) + 1 if len(unique_ids) > 0 else 20)
    for cid in unique_ids:
        if cid == 0:
            continue  # skip background
        rgba = cmap(cid)
        overlay[seg_map == cid] = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

    blended = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(blended)