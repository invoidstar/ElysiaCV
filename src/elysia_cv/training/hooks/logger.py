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
    - ``after_run`` prints a "Training Complete" summary box with best accuracy.

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
        ema_alpha: float = 0.0,
        message_hub: Any = None,
    ) -> None:
        self.log_freq = log_freq
        self.total_steps = total_steps
        self.ema_alpha = ema_alpha
        self._start_time = time.time()
        self._is_tty = sys.stdout.isatty()
        self._pbar = None
        self._loss_window: list[float] = []
        self._loss_ema: float | None = None  # EMA-smoothed loss
        self._best_acc: float | None = None  # tracked from EvalHook
        self._final_ckpt: str = ""  # tracked from CheckpointHook
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

        # MessageHub: write step-level metrics for other hooks to read
        self._write_to_hub(step=0, loss=0.0, lr=0.0)

    def _write_to_hub(self, **kwargs) -> None:
        """Publish scalars to the global MessageHub (best-effort, no crash)."""
        try:
            from elysia_cv.training.message_hub import get_message_hub
            hub = get_message_hub()
            for k, v in kwargs.items():
                hub.update_scalar(f"train/{k}", float(v))
        except Exception:
            pass

    def after_step(self, loop) -> None:
        if not self._is_main:
            return

        step = loop.state.global_step
        loss = float(loop.state.last_loss) if loop.state.last_loss is not None else 0.0
        lr = loop.optimizer.param_groups[0].get("lr", 0.0)

        self._loss_window.append(loss)

        # EMA smoothing (if enabled)
        if self.ema_alpha > 0:
            if self._loss_ema is None:
                self._loss_ema = loss
            else:
                self._loss_ema = self.ema_alpha * loss + (1 - self.ema_alpha) * self._loss_ema

        # Display value: EMA if enabled, else window average
        if self.ema_alpha > 0 and self._loss_ema is not None:
            display_loss = self._loss_ema
        elif len(self._loss_window) >= self.log_freq:
            display_loss = sum(self._loss_window[-self.log_freq:]) / self.log_freq
        else:
            display_loss = loss

        # ── TTY mode: update progress bar ─────────────────────
        if self._pbar is not None:
            self._pbar.update(1)
            if step % self.log_freq == 0 or step <= 5:
                self._pbar.set_postfix(
                    loss=f"{display_loss:.4f}", lr=f"{lr:.2e}", eta=self._eta(step),
                )
            return

        # ── Non-TTY mode: timestamped percentage lines ──────────
        if step % self.log_freq == 0 or step == 1:
            pct = 100.0 * step / max(self.total_steps, 1)
            print(
                f"  [{self._timestamp()}] [{pct:5.1f}%] step {step:>6,d}/{self.total_steps:,d}"
                f"  loss={display_loss:.4f}  lr={lr:.2e}  ETA {self._eta(step)}",
                flush=True,
            )

    def after_run(self, loop) -> None:
        if not self._is_main:
            return
        if self._pbar is not None:
            self._pbar.close()

        step = loop.state.global_step
        elapsed = time.time() - self._start_time
        print(
            f"\n  [{self._timestamp()}] Training finished in {_format_time(elapsed)}"
            f"  ({step:,} steps)",
            flush=True,
        )

        # ── Final summary box ──────────────────────────────────
        W = 62
        print("", flush=True)
        print("╔" + "═" * W + "╗", flush=True)
        print("║" + "  Training Complete".ljust(W) + "║", flush=True)
        print("║" + f"  Steps:      {step:,}".ljust(W) + "║", flush=True)
        if self._best_acc is not None:
            print("║" + f"  Best Acc:   {self._best_acc:.4f}".ljust(W) + "║", flush=True)
        if self._final_ckpt:
            ckpt = self._final_ckpt.split("/")[-2] + "/" + self._final_ckpt.split("/")[-1] if "/" in self._final_ckpt else self._final_ckpt
            print("║" + f"  Checkpoint: {ckpt}".ljust(W) + "║", flush=True)
        print("╚" + "═" * W + "╝", flush=True)

    # ── Callbacks for other hooks to relay messages ────────────

    def on_eval_result(
        self,
        step: int,
        acc: float = 0.0,
        val_loss: float = 0.0,
        improved: bool = False,
        results: dict | None = None,
        primary: float = 0.0,
    ) -> None:
        """Record eval result — called by EvalHook. Prints immediately.

        Supports both legacy ``(acc, val_loss)`` and new ``(results, primary)``
        keyword signatures.
        """
        # Best tracking
        use_acc = acc if acc > 0 else primary
        if self._best_acc is None or use_acc > self._best_acc:
            self._best_acc = use_acc

        flag = " *" if improved else ""

        if results and len(results) > 1:
            # Multi-metric results (detection style)
            parts = "  ".join(f"{k}={v:.4f}" for k, v in results.items())
            msg = f"  [eval]  step {step:>6,d}  {parts}{flag}"
        else:
            msg = f"  [eval]  step {step:>6,d}  acc={use_acc:.4f}{flag}"

        if self._pbar is not None:
            self._pbar.write(msg)
        else:
            print(msg, flush=True)

    def on_checkpoint_saved(self, name: str, path: str = "") -> None:
        """Record checkpoint save — called by CheckpointHook. Prints immediately."""
        if "final" in name:
            self._final_ckpt = path
        msg = f"  [checkpoint] {name}"
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