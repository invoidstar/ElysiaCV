"""Data transforms — registered via ``elysia_cv.core.registry.TRANSFORMS``.

Transforms are composed from YAML config via ``build_pipeline``.
Each transform is a callable registered with ``@TRANSFORMS.register_module()``.
"""

from __future__ import annotations

# Trigger module-level @TRANSFORMS.register_module() decorators
from elysia_cv.data.transforms.loading import (  # noqa: F401
    LoadImageFromFile,
    LoadAnnotations,
)
from elysia_cv.data.transforms.processing import (  # noqa: F401
    Resize,
    RandomResizedCrop,
    CenterCrop,
    Normalize,
)
from elysia_cv.data.transforms.augmentation import (  # noqa: F401
    RandomFlip,
    ColorJitter,
    RandomErasing,
    PhotoMetricDistortion,
    MixUp,
    CutMix,
    RandomRotation,
    RandomAffine,
    Mosaic,
)
from elysia_cv.data.transforms.formatting import (  # noqa: F401
    PackInputs,
    PackDetInputs,
    PackSegInputs,
)


class Compose:
    """A composed transform pipeline with readable ``__repr__``.

    Wraps a list of callable transforms.  Similar to ``torchvision.transforms.Compose``
    but designed for the MMDetection-style ``results`` dict protocol.

    Supports both per-sample transforms and batch-level transforms (e.g. MixUp).
    Batch transforms are applied after collation, on the full batch dict.

    Example repr::

        Compose(
          LoadImageFromFile
          → LoadAnnotations(with_bbox=True)
          → Resize(scale=(1333, 800), keep_ratio=True)
          → Normalize
          → PackDetInputs
        )
    """

    def __init__(self, transforms: list, batch_transforms: list | None = None) -> None:
        self.transforms = transforms
        self.batch_transforms = batch_transforms or []

    def __call__(self, results: dict) -> dict:
        for t in self.transforms:
            results = t(results)
        return results

    def apply_batch(self, batch: dict) -> dict:
        """Apply batch-level transforms (e.g. MixUp) to a collated batch dict."""
        for t in self.batch_transforms:
            batch = t(batch)
        return batch

    def __repr__(self) -> str:
        fmt = (
            f"{type(self).__module__}.{type(self).__qualname__}"
            f"(\n"
        )
        for t in self.transforms:
            fmt += f"  {type(t).__name__}"
            # Include key init params for common transforms
            extra = _repr_params(t)
            if extra:
                fmt += f"({extra})"
            fmt += "\n"
        fmt += ")"
        return fmt

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int):
        return self.transforms[idx]

    def __iter__(self):
        return iter(self.transforms)


def _repr_params(obj: object) -> str:
    """Extract a short descriptive string from a transform's key attributes."""
    try:
        attrs: dict[str, object] = {}
        for k in ("size", "scale", "keep_ratio", "mean", "with_bbox", "num_classes",
                  "prob", "topk", "input_size"):
            v = getattr(obj, k, None)
            if v is not None and not k.startswith("_"):
                if isinstance(v, (list, tuple)) and len(v) <= 4:
                    attrs[k] = v
                elif isinstance(v, (int, float, bool, str)):
                    attrs[k] = v
        return ", ".join(f"{k}={v}" for k, v in attrs.items())
    except Exception:
        return ""


def build_pipeline(cfg_list: list[dict]) -> Compose:
    """Build a transform pipeline from a list of config dicts.

    Each dict MUST contain a ``"type"`` key.
    Returns a :class:`Compose` that prints a readable pipeline summary.
    """
    from elysia_cv.core.registry import TRANSFORMS

    transforms = []
    for cfg in cfg_list:
        transforms.append(TRANSFORMS.build(cfg))
    return Compose(transforms)


def build_batch_pipeline(cfg_list: list[dict] | None) -> list:
    """Build batch-level transforms (e.g. MixUp) from config dicts.

    Returns a list of callables; pass to :meth:`Compose.apply_batch`.
    """
    from elysia_cv.core.registry import TRANSFORMS

    if not cfg_list:
        return []
    batch_tfs = []
    for cfg in cfg_list:
        batch_tfs.append(TRANSFORMS.build(cfg))
    return batch_tfs