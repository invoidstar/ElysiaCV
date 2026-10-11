"""Lazy import utility — delays heavy imports until first use.

Extracted from ElysiaRobot's policy contract pattern.  Used widely
(27+ call sites in ElysiaRobot) to defer torch / transformers imports
for dry-run and preflight paths.

Usage::

    from elysia_cv.utils.lazy_import import load_callable_target

    runtime_cls = load_callable_target("elysia_cv.models.classification.resnet.runtime:ResNetRuntime")
"""

from __future__ import annotations

import importlib
from typing import Any


def load_callable_target(target: str) -> Any:
    """Import and return a callable from a ``module.attr`` string.

    Parameters
    ----------
    target : str
        Fully-qualified path in the form ``"package.module:attr"``.

    Returns
    -------
    Any
        The imported callable.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the attribute does not exist on the module.
    """
    if ":" not in target:
        raise ValueError(
            f"target must be in 'module:attr' format, got {target!r}"
        )
    module_name, attr_name = target.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"failed to import module {module_name!r} for target {target!r}"
        ) from exc
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"module {module_name!r} has no attribute {attr_name!r}"
        )
    return getattr(module, attr_name)


def load_module_attr(target: str) -> Any:
    """Like :func:`load_callable_target` but does not require the target to be callable."""
    return load_callable_target(target)