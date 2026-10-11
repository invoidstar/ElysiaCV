"""Core plugin-layer error types.

Directly adapted from ElysiaRobot core/errors.py — the exception
hierarchy is fully generic and has zero domain dependencies.
"""

from __future__ import annotations


class ElysiaCoreError(Exception):
    """Base error for the Core API."""


class PluginError(ElysiaCoreError):
    """Plugin loading or protocol validation failed."""


class PluginApiVersionError(PluginError):
    """Plugin API version is incompatible with the current Core."""


class PluginDiscoveryError(PluginError):
    """Plugin module discovery or import failed."""