"""EarlyStoppingHook — stop training when a monitored metric stops improving.

Integrates with :class:`EvalHook` via the ``on_eval_result`` callback
(already broadcast to all hooks by EvalHook's ``_notify_logger``).

Usage in YAML::

    eval:
      eval_freq: 5000
      early_stop:
        metric: Accuracy
        mode: max
        patience: 5
        min_delta: 0.001
"""

from __future__ import annotations

from typing import Any

from elysia_cv.training.hooks.base import Hook


class EarlyStoppingHook(Hook):
    """Stop training early when a monitored metric stops improving.

    Monitors a single eval metric (e.g. ``Accuracy``, ``mAP``, ``Loss``).
    If no improvement exceeds ``min_delta`` for ``patience`` consecutive
    eval cycles, sets ``loop.state.should_stop = True``.

    Args:
        metric: Which eval result key to monitor (must match a key in
            EvalHook's result dict).
        mode: ``"max"`` (higher is better, e.g. Accuracy, mAP) or
            ``"min"`` (lower is better, e.g. Loss).
        patience: How many eval cycles to wait before stopping.
        min_delta: Minimum absolute change to count as an improvement.
    """

    def __init__(
        self,
        metric: str = "Accuracy",
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"EarlyStopping mode must be 'max' or 'min', got '{mode}'")
        self.metric = metric
        self.mode = mode
        self.patience = max(patience, 1)
        self.min_delta = float(min_delta)

        self._best: float | None = None
        self._wait: int = 0
        self._loop: Any = None
        self._stopped_epoch: int = 0

    # ── lifecycle ──────────────────────────────────────────────

    def before_run(self, loop: Any) -> None:
        """Capture loop reference so we can set ``should_stop``."""
        self._loop = loop
        self._best = None
        self._wait = 0

    def on_eval_result(
        self,
        step: int,
        results: dict[str, float],
        primary: float,
        improved: bool,
        **kwargs: Any,
    ) -> None:
        """Called by EvalHook after each evaluation cycle.

        Checks whether the monitored metric has improved; if not,
        increments the patience counter and stops the loop when
        the threshold is reached.
        """
        value = results.get(self.metric)
        if value is None:
            # Monitored metric not in results yet — skip
            return

        if self._best is None:
            self._best = value
            self._wait = 0
            return

        if self._is_better(value, self._best):
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._loop.state.should_stop = True
                self._stopped_epoch = self._loop.state.epoch
                # Log reason via LoggerHook if available
                self._emit_message(
                    f"[EarlyStopping] {self.metric} did not improve for "
                    f"{self._wait} evals (best={self._best:.4f}), stopping at step {step}.",
                )

    def after_run(self, loop: Any) -> None:
        """Log final early-stopping status."""
        if self._stopped_epoch > 0:
            self._emit_message(
                f"[EarlyStopping] Training stopped early at epoch "
                f"{self._stopped_epoch} (best {self.metric}={self._best:.4f}).",
            )
        elif self._best is not None:
            self._emit_message(
                f"[EarlyStopping] Patience not exceeded "
                f"(best {self.metric}={self._best:.4f} after {self._wait} stagnant evals).",
            )

    # ── helpers ─────────────────────────────────────────────────

    def _is_better(self, current: float, best: float) -> bool:
        """Return True if *current* is better than *best* by at least ``min_delta``."""
        delta = current - best if self.mode == "max" else best - current
        return delta > self.min_delta

    def _emit_message(self, msg: str) -> None:
        """Best-effort print; also try LoggerHook for structured output."""
        try:
            print(msg, flush=True)
        except Exception:
            pass
        if self._loop is not None:
            try:
                for h in self._loop.callback._hooks:
                    if hasattr(h, "log"):
                        h.log(msg)
                        break
            except Exception:
                pass