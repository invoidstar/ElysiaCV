"""Training loop state — adapted from ElysiaRobot training/loop_state.py.

Separates the training loop from backend differences:
- :class:`TrainerLoopContext`: few-changing config (plan, strategy, model, callbacks).
- :class:`TrainerLoopState`: continuously changing state (steps, cursor, loss).

Checkpoint / resume only reads the stable fields of ``TrainerLoopState``,
keeping the loop itself model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class NoopComponentsController:
    """Default no-op components controller.

    When no explicit role adapter is injected, the real native/accelerate
    engine will replace this with a proper adapter.
    """

    def apply_trainable_components(self, model: Any, plan: dict[str, Any], components: Any) -> None:
        plan["trainable_components"] = (
            components.to_dict() if hasattr(components, "to_dict") else components
        )


@dataclass
class TrainerLoopContext:
    """Few-changing configuration for one training run.

    Fine-grained role adapters are injected via *strategy*, *model*, *recipe*,
    and *callbacks*, each carrying a separate responsibility (backend, forward,
    objectives, event bus).
    """

    plan: dict[str, Any]
    strategy: Any
    model: Any
    recipe: Any
    callbacks: Any
    components: Any = field(default_factory=NoopComponentsController)
    device: Any | None = None
    distributed: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerLoopState:
    """Continuously changing state during training.

    *global_step* is the optimizer step count (checkpoint-compatible);
    *micro_step* is the total consumed micro-batch count;
    *epoch* / *data_cursor* describe the sampling cursor position.
    """

    global_step: int = 0
    micro_step: int = 0
    epoch: int = 0
    data_cursor: dict[str, Any] = field(default_factory=dict)
    last_loss: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False

    def stable_checkpoint_state(self) -> dict[str, Any]:
        """Return only the fields that should be persisted in checkpoints."""
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "epoch": self.epoch,
            "data_cursor": dict(self.data_cursor),
        }

    def restore_from_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore step / data cursor from a checkpoint.

        Only reads the fields declared by :meth:`stable_checkpoint_state`;
        *last_loss*, *metrics*, and *should_stop* are runtime transient.
        """
        self.global_step = int(state.get("global_step", self.global_step))
        self.micro_step = int(state.get("micro_step", self.micro_step))
        self.epoch = int(state.get("epoch", self.epoch))
        cursor = state.get("data_cursor")
        if isinstance(cursor, dict):
            self.data_cursor = dict(cursor)