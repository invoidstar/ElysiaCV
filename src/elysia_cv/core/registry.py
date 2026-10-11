"""Unified Registry for reusable CV components.

Inspired by MMEngine's Registry design, this provides a lightweight
name → class mapping with decorator-based registration.  Used for
shared components that are composed into models at config-time:
Backbones, Necks, Heads, Transforms, Losses, Metrics, Hooks.

Model plugins themselves use the separate Plugin discovery system
in ``core/plugin.py``.

Usage::

    from elysia_cv.core.registry import BACKBONES

    @BACKBONES.register_module()
    class ResNet(nn.Module):
        ...

    # later:
    backbone = BACKBONES.build({"type": "ResNet", "depth": 50})
"""

from __future__ import annotations

from typing import Any


class Registry:
    """A registry that maps string names to classes.

    Parameters
    ----------
    name : str
        Human-readable registry name (for error messages).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: dict[str, type] = {}

    @property
    def name(self) -> str:
        return self._name

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __len__(self) -> int:
        return len(self._module_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, items={len(self._module_dict)})"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_module(self, name: str | None = None, *, force: bool = False):
        """Decorator: register a class under *name* (or ``cls.__name__``).

        Parameters
        ----------
        name : str or None
            Registry key.  If None, uses the decorated class's ``__name__``.
        force : bool
            If True, silently overwrite an existing entry.
        """

        def _register(cls: type) -> type:
            key = name if name is not None else cls.__name__
            if key in self._module_dict and not force:
                existing = self._module_dict[key]
                raise KeyError(
                    f"{key!r} is already registered in {self._name} registry "
                    f"(existing: {existing.__module__}.{existing.__qualname__}). "
                    f"Use force=True to overwrite."
                )
            self._module_dict[key] = cls
            return cls

        return _register

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, key: str) -> type:
        """Return the class registered under *key*."""
        if key not in self._module_dict:
            raise KeyError(
                f"{key!r} not found in {self._name} registry. "
                f"Available: {sorted(self._module_dict.keys())}"
            )
        return self._module_dict[key]

    def keys(self) -> list[str]:
        return list(self._module_dict.keys())

    # ------------------------------------------------------------------
    # Build-from-config
    # ------------------------------------------------------------------

    def build(self, cfg: dict[str, Any] | None, **overrides: Any) -> Any:
        """Build an instance from a dict config.

        The dict MUST contain a ``"type"`` key identifying the registered class.
        All other keys are passed as keyword arguments to the constructor.
        """
        if cfg is None:
            return None
        cfg = dict(cfg)  # shallow copy so we can pop
        cfg.update(overrides)
        type_name = cfg.pop("type", None)
        if type_name is None:
            raise ValueError(
                f"config dict for {self._name} registry missing required key 'type'. "
                f"Got keys: {list(cfg.keys())}"
            )
        cls = self.get(type_name)
        return cls(**cfg)


# ---------------------------------------------------------------------------
# Global Registry instances — one per component category
# ---------------------------------------------------------------------------

BACKBONES = Registry("backbone")
"""Shared vision backbones (ResNet, ViT, Swin, ...)."""

NECKS = Registry("neck")
"""Feature-fusion necks (FPN, Identity, ...)."""

HEADS = Registry("head")
"""Task-specific heads (ClsHead, DetHead, SegHead, ...)."""

MODELS = Registry("model")
"""Composed task-models (ImageClassifier, ImageDetector, ...).  Note: model
*plugins* use the separate Plugin discovery system; this registry is for
models assembled from shared components."""

TRANSFORMS = Registry("transform")
"""Data transforms (Resize, RandomFlip, Normalize, ...)."""

LOSSES = Registry("loss")
"""Loss functions (CrossEntropyLoss, FocalLoss, IoULoss, ...)."""

METRICS = Registry("metric")
"""Evaluation metrics (Accuracy, mAP, mIoU, ...)."""

HOOKS = Registry("hook")
"""Training hooks (CheckpointHook, LoggerHook, EvalHook, ...)."""

STRATEGIES = Registry("strategy")
"""Distributed training strategies (DDP, FSDP, ...)."""

OPTIMIZERS = Registry("optimizer")
"""Optimizer builders (SGD, AdamW, ...)."""

SCHEDULERS = Registry("scheduler")
"""LR scheduler builders (MultiStepLR, CosineAnnealing, ...)."""