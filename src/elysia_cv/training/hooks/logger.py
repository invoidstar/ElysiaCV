"""LoggerHook — log training metrics to stdout and optionally W&B."""

from __future__ import annotations

import time
from typing import Any

from elysia_cv.training.hooks.base import Hook


class LoggerHook(Hook):
    """Log training progress at regular step intervals."""

    def __init__(
        self,
        log_freq: int = 50,
        log_header_interval: int = 50,
        wandb_enabled: bool = False,
    ) -> None:
        self.log_freq = log_freq
        self.log_header_interval = log_header_interval
        self.wandb_enabled = wandb_enabled
        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._header_count = 0

    def after_step(self, loop) -> None:
        step = loop.state.global_step
        if step <= 0 or step % self.log_freq != 0:
            return

        now = time.time()
        elapsed = now - self._start_time
        step_time = (now - self._last_log_time) / max(self.log_freq, 1)

        # Header
        if self._header_count % self.log_header_interval == 0:
            print(f"{'Step':>8s} {'Loss':>10s} {'LR':>10s}  {'Time/step':>10s}  {'Elapsed':>10s}")
        self._header_count += 1

        loss = loop.state.last_loss
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
        lr = loop.optimizer.param_groups[0].get("lr", 0.0)
        elapsed_str = _format_time(elapsed)

        print(f"{step:8d} {loss_str:>10s} {lr:>10.6f}  {step_time:>8.3f}s  {elapsed_str:>10s}")

        self._last_log_time = now

    def after_run(self, loop) -> None:
        elapsed = time.time() - self._start_time
        print(f"\nTraining finished.  Total time: {_format_time(elapsed)}")


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m{secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins}m{secs}s"