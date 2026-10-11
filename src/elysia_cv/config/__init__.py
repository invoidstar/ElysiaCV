"""Configuration system — loading, validation, and typed access.

Public API
----------
``load_cv_config``  — load YAML → resolved dict
``load_cv_config_with_object`` — load YAML → typed :class:`CVConfig`
``CVConfig`` — frozen dataclass with ``from_mapping`` + ``validate``
"""

from __future__ import annotations

from elysia_cv.config.config import CVConfig
from elysia_cv.config.loader import load_cv_config, load_cv_config_with_object

__all__ = [
    "CVConfig",
    "load_cv_config",
    "load_cv_config_with_object",
]