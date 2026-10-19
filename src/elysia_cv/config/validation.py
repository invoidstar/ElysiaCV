"""Cross-section config consistency validation.

Validates relationships BETWEEN sections that can't be checked at the
section-dataclass level.  Separated from config.py to keep the core
dataclass lean.
"""

from __future__ import annotations

from elysia_cv.config.config import CVConfig


_DETECTION_MODELS = {"fcos"}
_SEGMENTATION_MODELS = {"fcn"}


def validate_cross_section(config: CVConfig) -> list[str]:
    """Validate cross-section consistency; return a list of error messages.

    An empty list means the config is valid.  This is intentionally a
    separate pass so it can be called from preflight / dry-run tools
    without importing heavy model / dataset code.
    """
    errors: list[str] = []

    model_type = config.model.type

    # ── train stage checks ──
    if config.run.stage == "train":
        if config.schedule.steps < 100:
            errors.append(
                f"schedule.steps={config.schedule.steps} seems too small for training"
            )
        if config.schedule.warmup_steps >= config.schedule.steps:
            errors.append(
                f"schedule.warmup_steps ({config.schedule.warmup_steps}) >= "
                f"schedule.steps ({config.schedule.steps})"
            )

    # ── test stage checks ──
    if config.run.stage == "test":
        if not config.eval.metrics:
            errors.append("eval.metrics is required for test stage")

    # ── model pretrained_path existence ──
    if config.model.pretrained_path:
        from pathlib import Path
        p = Path(config.model.pretrained_path)
        if p.exists() and not p.is_file():
            errors.append(f"model.pretrained_path is not a file: {p}")

    # ── Detection-specific validation ──
    if model_type in _DETECTION_MODELS:
        head = getattr(config.model, "head", None)
        if head is not None:
            num_classes = _get_cfg_value(head, "num_classes", 0)
            if num_classes <= 0:
                errors.append(
                    f"Detection model '{model_type}' requires "
                    f"head.num_classes > 0 (got {num_classes})"
                )
        # FPN neck must have in_channels matching backbone out_channels
        neck = getattr(config.model, "neck", None)
        backbone = getattr(config.model, "backbone", None)
        if neck is not None and backbone is not None:
            neck_in = _get_cfg_value(neck, "in_channels", [])
            bb_depth = _get_cfg_value(backbone, "depth", 0)
            if bb_depth and neck_in:
                _check_fpn_backbone_compat(errors, bb_depth, neck_in, model_type)

    # ── Segmentation-specific validation ──
    if model_type in _SEGMENTATION_MODELS:
        head = getattr(config.model, "head", None)
        if head is not None:
            num_classes = _get_cfg_value(head, "num_classes", 0)
            if num_classes <= 1:
                errors.append(
                    f"Segmentation model '{model_type}' requires "
                    f"head.num_classes > 1 (got {num_classes})"
                )

    return errors


def _get_cfg_value(obj: object, key: str, default: object = None) -> object:
    """Read a value from either a dict or a frozen dataclass field."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default) if hasattr(obj, key) else default


def _check_fpn_backbone_compat(
    errors: list[str],
    depth: int,
    neck_in_channels: list[int],
    model_type: str,
) -> None:
    """Warn if FPN in_channels don't match canonical ResNet depths."""
    canonical: dict[int, list[int]] = {
        18:  [64, 64, 128, 256],
        34:  [64, 128, 256, 512],
        50:  [256, 512, 1024, 2048],
        101: [256, 512, 1024, 2048],
    }
    expected = canonical.get(depth)
    if expected is None:
        return  # unknown depth, skip check
    if neck_in_channels != expected:
        errors.append(
            f"FPN neck.in_channels={neck_in_channels} may not match "
            f"ResNet-{depth} expected channels={expected} "
            f"(verify out_indices alignment)"
        )


def validate_or_raise(config: CVConfig) -> None:
    """Like :func:`validate_cross_section`, but raises on first error."""
    errs = validate_cross_section(config)
    if errs:
        raise ValueError("config validation failed:\n  " + "\n  ".join(errs))