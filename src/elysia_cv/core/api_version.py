"""Core plugin API version definition.

Analogous to ElysiaRobot's core/api_version.py. The single version string
governs plugin compatibility: major must match, minor may extend
backward-compatibly.
"""

from __future__ import annotations

CORE_API_VERSION = "1.0"
"""Current stable plugin-protocol version.

Rules: major version MUST match; minor version can extend
backward-compatibly.
"""


def api_major(version: str) -> int:
    """Return the major part of a semantic version; -1 for invalid input."""
    try:
        return int(str(version).split(".", 1)[0])
    except (TypeError, ValueError):
        return -1


def is_compatible_api_version(version: str, *, core_version: str = CORE_API_VERSION) -> bool:
    """Return True when *version* is compatible with *core_version*."""
    return api_major(version) == api_major(core_version)