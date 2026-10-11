"""Hook event bus — lightweight event system for training lifecycle callbacks.

Hooks register at specific events (before/after_step, before/after_epoch, ...)
and the engine fires those events during training.  This is the central
extensibility point: logging, checkpointing, evaluation, scheduling are all
implemented as hooks.
"""

from __future__ import annotations

from typing import Any


class Hook:
    """Base hook class. Override any method to hook into the training loop.

    All methods are no-ops by default.
    """

    def before_run(self, loop: "TrainerLoop") -> None:
        """Called once before the training loop starts."""

    def after_run(self, loop: "TrainerLoop") -> None:
        """Called once after the training loop finishes."""

    def before_epoch(self, loop: "TrainerLoop") -> None:
        """Called at the start of each epoch."""

    def after_epoch(self, loop: "TrainerLoop") -> None:
        """Called at the end of each epoch."""

    def before_step(self, loop: "TrainerLoop") -> None:
        """Called before each optimizer step."""

    def after_step(self, loop: "TrainerLoop") -> None:
        """Called after each optimizer step."""

    def before_micro_step(self, loop: "TrainerLoop") -> None:
        """Called before each micro-batch forward/backward."""

    def after_micro_step(self, loop: "TrainerLoop") -> None:
        """Called after each micro-batch forward/backward."""


class HookBus:
    """Container that fires events to all registered hooks."""

    def __init__(self) -> None:
        self._hooks: list[Hook] = []

    def register(self, hook: Hook) -> None:
        self._hooks.append(hook)

    def __len__(self) -> int:
        return len(self._hooks)

    def _fire(self, event: str, loop: "TrainerLoop") -> None:
        for hook in self._hooks:
            getattr(hook, event, lambda _: None)(loop)

    def before_run(self, loop: "TrainerLoop") -> None:
        self._fire("before_run", loop)

    def after_run(self, loop: "TrainerLoop") -> None:
        self._fire("after_run", loop)

    def before_epoch(self, loop: "TrainerLoop") -> None:
        self._fire("before_epoch", loop)

    def after_epoch(self, loop: "TrainerLoop") -> None:
        self._fire("after_epoch", loop)

    def before_step(self, loop: "TrainerLoop") -> None:
        self._fire("before_step", loop)

    def after_step(self, loop: "TrainerLoop") -> None:
        self._fire("after_step", loop)

    def before_micro_step(self, loop: "TrainerLoop") -> None:
        self._fire("before_micro_step", loop)

    def after_micro_step(self, loop: "TrainerLoop") -> None:
        self._fire("after_micro_step", loop)