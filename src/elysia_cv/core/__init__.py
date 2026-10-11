"""ElysiaCV core — lightweight plugin protocol, registry, and version contract.

Only depends on the Python standard library. Heavy imports (torch, transformers, etc.)
are deferred through lazy import patterns so that read-only workflows (doctor, dry-run,
preflight) never trigger model loading.
"""

from __future__ import annotations

from elysia_cv.core.api_version import CORE_API_VERSION

__all__ = ["CORE_API_VERSION"]