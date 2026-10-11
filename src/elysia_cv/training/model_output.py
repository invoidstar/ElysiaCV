"""Unified model forward output contract — adapted from ElysiaRobot.

``ModelOutput`` is the stable data boundary between a model's forward pass
and the training engine's objective / metric computation.

*losses* is an explicit multi-objective decomposition: keys are objective
``loss_key`` strings, values are *unweighted* raw loss components.
*loss* is the scalar total loss reported by the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelOutput:
    """Unified output from a model forward pass.

    Attributes
    ----------
    loss : Any
        The scalar total loss reported by the model.
    losses : dict[str, Any]
        Per-objective loss components (unweighted).
    loss_masks : dict[str, Any] or None
        Per-sample masks / weights keyed by loss_key.
    metrics : dict[str, Any]
        Scalar / tensor metrics to log with this step.
    """

    loss: Any
    losses: dict[str, Any] = field(default_factory=dict)
    loss_masks: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)