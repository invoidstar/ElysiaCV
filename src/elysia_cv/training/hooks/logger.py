"""LoggerHook — TTY-aware training progress logging with tqdm bar, timestamps, and ETA."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from typing import Any

from elysia_cv.training.hooks.base import Hook
from elysia_cv.training.distributed import unwrap_model


class LoggerHook(Hook):
    """Log training progress with TTY-aware output, timestamp, and ETA.

    - **TTY mode** (interactive terminal): renders a compact ``tqdm`` progress bar
      with built-in ``{elapsed}`` / ``{remaining}``.
    - **Non-TTY mode** (piped / tee): prints timestamped lines every ``log_freq`` steps
      with estimated remaining time.
    - Total parameter count and GPU name are shown once in ``before_run``.

    Parameters
    ----------
    log_freq : int
        Log every N optimizer steps.
    total_steps : int
        Total number of optimizer steps (for the progress bar).
    """

    def __init__(
        self,
        log_freq: int = 50,
        total_steps: int = 1,
    ) -> None:
        self.log_freq = log_freq
        self.total_steps = total_steps
        self._start_time = time.time()
        self._is_tty = sys.stdout.isatty()
        self._pbar = None
        self._loss_window: list[float] = []
        self._result_lines: list[str] = []  # accumulated for final print
        self._is_main = self._should_log()

        # Try to import tqdm for TTY mode
        if self._is_tty:
            try:
                from tqdm import tqdm  # noqa: F401
                self._tqdm_available = True
            except ImportError:
                self._tqdm_available = False
        else:
            self._tqdm_available = False

    # ── Helpers ─────────────────────────────────────────────────

    def _should_log(self) -> bool:
        """Return True if this rank should print (rank 0 in DDP, always in non-DDP)."""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        return True

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _eta(self, step: int) -> str:
        """Estimated remaining time based on elapsed wall time."""
        elapsed = time.time() - self._start_time
        if step <= 0 or elapsed <= 0:
            return "---"
        total_eta = elapsed / step * (self.total_steps - step)
        return _format_time(total_eta)

    # ── Hook lifecycle ──────────────────────────────────────────

    def before_run(self, loop) -> None:
        if self._is_tty and self._tqdm_available:
            from tqdm import tqdm
            self._pbar = tqdm(
                total=self.total_steps, desc="Training", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                dynamic_ncols=True,
                postfix={"loss": "---", "lr": "---", "eta": "---"},
            )

    def after_step(self, loop) -> None:
        if not self._is_main:
            return

        step = loop.state.global_step
        loss = loop.state.last_loss
        lr = loop.optimizer.param_groups[0].get("lr", 0.0)

        self._loss_window.append(float(loss) if loss is not None else 0.0)

        # ── TTY mode: update progress bar ─────────────────────
        if self._pbar is not None:
            self._pbar.update(1)
            if step % self.log_freq == 0 and len(self._loss_window) >= self.log_freq:
                avg = sum(self._loss_window[-self.log_freq:]) / self.log_freq
                self._pbar.set_postfix(
                    loss=f"{avg:.4f}", lr=f"{lr:.2e}", eta=self._eta(step),
                )
            elif step <= 5:
                self._pbar.set_postfix(
                    loss=f"{loss:.4f}", lr=f"{lr:.2e}", eta=self._eta(step),
                )
            return

        # ── Non-TTY mode: timestamped percentage lines ──────────
        if step % self.log_freq == 0 or step == 1:
            window = self._loss_window[-self.log_freq:]
            avg = sum(window) / max(len(window), 1)
            pct = 100.0 * step / max(self.total_steps, 1)
            print(
                f"  [{self._timestamp()}] [{pct:5.1f}%] step {step:>6,d}/{self.total_steps:,d}"
                f"  loss={avg:.4f}  lr={lr:.2e}  ETA {self._eta(step)}",
                flush=True,
            )

    def after_run(self, loop) -> None:
        if not self._is_main:
            return
        if self._pbar is not None:
            self._pbar.close()
        elapsed = time.time() - self._start_time
        print(
            f"\n  [{self._timestamp()}] Training finished in {_format_time(elapsed)}"
            f"  ({loop.state.global_step:,} steps)",
            flush=True,
        )
        # Print accumulated results (from EvalHook / CheckpointHook)
        for line in self._result_lines:
            print(line, flush=True)

    # ── Callbacks for other hooks to relay messages ────────────

    def on_eval_result(self, step: int, acc: float, val_loss: float, improved: bool) -> None:
        """Record eval result for final summary."""
        flag = " *" if improved else ""
        msg = f"  [eval]  step {step:>6,d}  acc={acc:.4f}  val_loss={val_loss:.4f}{flag}"
        self._result_lines.append(msg)
        if self._pbar is not None:
            self._pbar.write(msg)
        else:
            print(msg, flush=True)

    def on_checkpoint_saved(self, name: str) -> None:
        """Record checkpoint save."""
        msg = f"  [checkpoint] {name}"
        self._result_lines.append(msg)
        if self._pbar is not None:
            self._pbar.write(msg)
        else:
            print(msg, flush=True)


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    if mins < 60:
        return f"{mins}m{secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h{mins}m{secs}s"