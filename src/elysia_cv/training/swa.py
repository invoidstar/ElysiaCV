"""SWA (Stochastic Weight Averaging) — running average of model snapshots.

Unlike EMA which smooths every step, SWA takes periodic snapshots during
the tail of training and averages them.  This captures the flat-minima
effect without the short-term noise smoothing of EMA.

Reference: "Averaging Weights Leads to Wider Optima and Better Generalization"
           (Izmailov et al., UAI 2018)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SWA:
    """Stochastic Weight Averaging via running-mean of periodic snapshots.

    Usage::

        swa = SWA(model, start_step=9000, interval=500)
        # In hook, once per optimizer step:
        swa.update(model)          # takes snapshot if it's time
        # Before evaluation:
        swa.apply_shadow(model)    # swap to averaged weights
        # After evaluation:
        swa.restore(model)         # swap back
        # Optional — at end of training:
        swa.merge_into(model)      # permanently replace weights
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        start_step: int = 9000,
        interval: int = 500,
    ) -> None:
        self.start_step = start_step
        self.interval = max(interval, 1)
        self.n_snapshots: int = 0
        self._step: int = 0

        # Running-average parameters
        self.avg_params: dict[str, torch.Tensor] = {}
        # Backup for apply/restore
        self._backup: dict[str, torch.Tensor] = {}

    # ── Update ─────────────────────────────────────────────────

    def update(self, model: nn.Module) -> None:
        """Take a snapshot (if interval passed) and update the running average.

        Call once per optimizer step **after** ``optimizer.step()``.
        """
        self._step += 1
        if self._step < self.start_step:
            return
        if (self._step - self.start_step) % self.interval != 0:
            return

        self.n_snapshots += 1
        n = self.n_snapshots
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if n == 1:
                    self.avg_params[name] = param.data.clone()
                else:
                    # Running mean: avg' = (n-1)/n * avg + 1/n * current
                    self.avg_params[name].mul_((n - 1.0) / n).add_(
                        param.data, alpha=1.0 / n
                    )

    # ── Apply / Restore ─────────────────────────────────────────

    def apply_shadow(self, model: nn.Module) -> None:
        """Temporarily swap model weights with SWA average (for evaluation).

        Must be followed by :meth:`restore`.
        """
        if self.n_snapshots == 0:
            return
        for name, param in model.named_parameters():
            if name in self.avg_params:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.avg_params[name])

    def restore(self, model: nn.Module) -> None:
        """Restore training weights after :meth:`apply_shadow`."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup.pop(name))

    def merge_into(self, model: nn.Module) -> None:
        """Permanently replace model weights with SWA average.

        No-op if no snapshots have been taken yet.
        """
        if self.n_snapshots == 0:
            return
        for name, param in model.named_parameters():
            if name in self.avg_params:
                param.data.copy_(self.avg_params[name])

    # ── Properties ──────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """True once at least one snapshot has been collected."""
        return self.n_snapshots > 0


class SWAHook:
    """Training hook that updates SWA after each optimizer step.

    Integrates with :class:`HookBus`::

        swa = SWA(model, start_step=9000, interval=500)
        hooks.register(SWAHook(swa))
    """

    def __init__(self, swa: SWA) -> None:
        self.swa = swa

    def after_step(self, loop: Any) -> None:
        """Update SWA with latest weights (takes snapshot if it's time)."""
        self.swa.update(loop.model)

    def after_run(self, loop: Any) -> None:
        """Merge SWA weights into model at end of training."""
        if self.swa.active:
            self.swa.merge_into(loop.model)