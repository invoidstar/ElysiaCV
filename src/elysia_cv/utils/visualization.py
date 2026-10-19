"""Detection & segmentation visualization utilities.

Draw bounding boxes, instance masks, and semantic segmentation overlays
onto images for debugging and qualitative analysis.
"""

from __future__ import annotations

from pathlib import Path
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


# ── PR curves ─────────────────────────────────────────────────


def plot_pr_curves(
    pr_data: dict[str, list[list[float]]],
    output_path: str | None = None,
    *,
    class_names: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 150,
) -> str | None:
    """Plot per-class Precision-Recall curves from MeanAP data.

    Args:
        pr_data: Dict from ``MeanAP.get_pr_data()``.
            Each value is ``[precisions_list, recalls_list]``.
        output_path: Path to save the PNG.  If ``None``, saves to
            ``analysis/pr_curves.png`` relative to CWD.
        class_names: Optional list mapping class ID → name.
        figsize: Figure size in inches.
        dpi: Output resolution.

    Returns:
        The absolute path to the saved PNG.
    """
    import os
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not pr_data:
        return None

    n_cls = len(pr_data)
    cols = min(4, n_cls)
    rows = (n_cls + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle("Per-Class Precision-Recall Curves (IoU=0.50)", fontsize=14, y=1.01)

    from elysia_cv.utils.visualization import _get_color

    for idx, (cls_id_str, (prec_vals, rec_vals)) in enumerate(sorted(pr_data.items(), key=lambda x: int(x[0]))):
        ax = axes[idx // cols][idx % cols]
        cls_id = int(cls_id_str)
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class {cls_id}"
        color = tuple(c / 255.0 for c in _get_color(cls_id))
        ax.plot(rec_vals, prec_vals, color=color, linewidth=1.5, label=name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_cls, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()

    if output_path is None:
        output_path = str(Path.cwd() / "analysis" / "pr_curves.png")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out.resolve())


# ── Detection comparison: GT vs predictions ──────────────────


def draw_gt_pred_comparison(
    image: Image.Image | np.ndarray,
    gt_bboxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_bboxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    *,
    class_names: list[str] | None = None,
    score_thr: float = 0.3,
    output_path: str | None = None,
) -> Image.Image:
    """Draw GT (green) and predicted (red) bounding boxes side-by-side.

    Produces a composite image with two panels:
      - Left:  original image + GT boxes in green
      - Right: original image + predicted boxes in red with scores

    Args:
        image: PIL Image or H×W×3 numpy array.
        gt_bboxes: N×4 GT boxes [x1,y1,x2,y2].
        gt_labels: length-N int array of ground-truth class IDs.
        pred_bboxes: P×4 predicted boxes.
        pred_scores: length-P confidence scores.
        pred_labels: length-P predicted class IDs.
        class_names: optional class name mapping.
        score_thr: only draw predictions with score ≥ threshold.
        output_path: if given, save to this path.

    Returns:
        PIL Image (side-by-side comparison).
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # Draw GT (green boxes)
    img_gt = image.copy()
    draw_gt = ImageDraw.Draw(img_gt)
    font_size = max(10, min(image.size) // 32)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i in range(len(gt_bboxes)):
        x1, y1, x2, y2 = map(int, gt_bboxes[i])
        label_id = int(gt_labels[i])
        color = (0, 180, 0)  # green
        draw_gt.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = class_names[label_id] if class_names and label_id < len(class_names) else f"cls{label_id}"
        draw_gt.text((x1 + 2, y1 + 2), f"GT {name}", fill=color, font=font)

    # Draw predictions (red boxes)
    img_pred = image.copy()
    draw_pred = ImageDraw.Draw(img_pred)

    for i in range(len(pred_bboxes)):
        if pred_scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = map(int, pred_bboxes[i])
        label_id = int(pred_labels[i])
        color = (220, 30, 30)  # red
        draw_pred.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = class_names[label_id] if class_names and label_id < len(class_names) else f"cls{label_id}"
        text = f"{name} {pred_scores[i]:.2f}"
        draw_pred.text((x1 + 2, y1 + 2), text, fill=color, font=font)

    # Composite: GT left, Pred right
    w = image.size[0]
    composite = Image.new("RGB", (w * 2 + 8, image.size[1]), color=(40, 40, 40))
    composite.paste(img_gt, (0, 0))
    composite.paste(img_pred, (w + 8, 0))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        composite.save(output_path)

    return composite