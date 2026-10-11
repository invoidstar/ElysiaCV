"""ElysiaCV — Unified training framework for computer vision research."""

from __future__ import annotations

__version__ = "0.1.0"


def _register_default_components() -> None:
    """Import all built-in component packages to trigger registry decorators.

    This is lazy: only called when a registry is first accessed or when
    ``elysia_cv`` is imported.  Plugin packages that have heavy dependencies
    (torch, etc.) are imported only when this function runs — not at
    top-level — so lightweight operations like ``elysia_cv.core.CORE_API_VERSION``
    stay fast.
    """
    import elysia_cv.components.backbones  # noqa: F401
    import elysia_cv.components.necks  # noqa: F401
    import elysia_cv.components.heads  # noqa: F401
    import elysia_cv.data.transforms  # noqa: F401
    import elysia_cv.losses  # noqa: F401
    import elysia_cv.metrics  # noqa: F401


# Trigger at import time so registries are populated for CLI / config use.
_register_default_components()