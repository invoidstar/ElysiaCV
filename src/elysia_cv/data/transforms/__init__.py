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
)
from elysia_cv.data.transforms.formatting import (  # noqa: F401
    PackInputs,
    PackDetInputs,
    PackSegInputs,
)


def build_pipeline(cfg_list: list[dict]) -> list:
    """Build a transform pipeline from a list of config dicts.

    Each dict MUST contain a ``"type"`` key.
    """
    from elysia_cv.core.registry import TRANSFORMS

    pipeline = []
    for cfg in cfg_list:
        pipeline.append(TRANSFORMS.build(cfg))
    return pipeline