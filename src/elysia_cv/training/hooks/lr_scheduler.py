"""LrSchedulerHook — step the LR scheduler."""

from __future__ import annotations

from elysia_cv.training.hooks.base import Hook


class LrSchedulerHook(Hook):
    """Step the LR scheduler after each optimizer step."""

    def after_step(self, loop) -> None:
        if hasattr(loop, "scheduler") and loop.scheduler is not None:
            loop.scheduler.step()