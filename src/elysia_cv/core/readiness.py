"""Plugin readiness levels.

Directly adapted from ElysiaRobot core/readiness.py. A plugin can expose
metadata, train in smoke tests, or serve evaluation without being
production-ready for formal results.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class PluginReadinessLevel(IntEnum):
    """Ordered readiness levels shared by model, dataset, and metric plugins."""

    metadata_only = 0
    trainable = 1
    inference_ready = 2
    evaluation_ready = 3
    production_ready = 4


READINESS_LEVEL_NAMES = tuple(level.name for level in PluginReadinessLevel)


def normalize_readiness_level(
    value: Any,
    *,
    default: PluginReadinessLevel = PluginReadinessLevel.metadata_only,
) -> PluginReadinessLevel:
    """Return a normalized readiness enum.

    *value* may already be a :class:`PluginReadinessLevel`, an integer value,
    or a case-insensitive string from ``READINESS_LEVEL_NAMES``.
    """
    if isinstance(value, PluginReadinessLevel):
        return value
    if value in (None, ""):
        return default
    if isinstance(value, int):
        return PluginReadinessLevel(value)
    text = str(value).strip().lower().replace("-", "_")
    try:
        return PluginReadinessLevel[text]
    except KeyError as exc:
        raise ValueError(f"unknown plugin readiness level: {value!r}") from exc


def readiness_level_name(value: Any) -> str:
    """Return a stable string name for a readiness value."""
    return normalize_readiness_level(value).name


def readiness_at_least(value: Any, minimum: Any) -> bool:
    """Return whether *value* satisfies *minimum*."""
    return normalize_readiness_level(value) >= normalize_readiness_level(minimum)