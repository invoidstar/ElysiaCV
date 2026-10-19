"""Model EMA (Exponential Moving Average) — shadow copy of model weights.

Maintains an exponentially smoothed copy of model parameters.  At evaluation
time, the EMA weights are temporarily swapped in, then restored.  This is a
well-established regularisation technique that typically adds +0.5–1.0%
accuracy with zero extra inference cost.

Reference: "Polyak averaging" / "Averaging Weights Leads to Wider Optima"
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of model weights.

    Updates the shadow copy every step::

        shadow_param = decay * shadow_param + (1 - decay) * current_param

    where ``decay`` ramps from ``start_decay`` to ``final_decay`` over
    ``warmup_steps``.

    Usage::

        ema = ModelEMA(model, decay=0.999)
        # After each optimizer.step():
        ema.update(model)
        # Before evaluation:
        ema.apply_shadow(model)
        # After evaluation:
        ema.restore(model)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.9999,
        start_decay: float | None = None,
        warmup_steps: int = 2000,
    ) -> None:
        self.decay = decay
        self.start_decay = start_decay if start_decay is not None else decay
        self.warmup_steps = max(warmup_steps, 1)
        self._step = 0

        # Deep-copy parameter buffers (detached from graph)
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._init_shadow(model)

    def _init_shadow(self, model: nn.Module) -> None:
        """Copy all trainable parameters into the shadow dict."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    # ── Current effective decay (ramp-up schedule) ──

    def _current_decay(self) -> float:
        """Cosine ramp from ``start_decay`` → ``decay`` over ``warmup_steps``."""
        if self._step >= self.warmup_steps:
            return self.decay
        progress = self._step / self.warmup_steps
        alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.start_decay + (self.decay - self.start_decay) * alpha

    # ── Update ───────────────────────────────────────────────

    def update(self, model: nn.Module) -> None:
        """Update the EMA shadow from the current model weights.

        Call once **after** ``optimizer.step()``.
        """
        self._step += 1
        d = self._current_decay()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)

    # ── Apply / Restore ──────────────────────────────────────

    def apply_shadow(self, model: nn.Module) -> None:
        """Temporarily swap model weights with EMA shadow (for evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore the original training weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup.pop(name))


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------


class EMAHook:
    """Training hook that updates ModelEMA after each optimizer step.

    Integrates with :class:`HookBus`::

        ema = ModelEMA(model)
        hooks.register(EMAHook(ema))
    """

    def __init__(self, ema: ModelEMA) -> None:
        self.ema = ema

    def after_step(self, loop: Any) -> None:
        """Update EMA shadow with latest weights."""
        self.ema.update(loop.model)

    # Compatibility with eval hook swap protocol
    @property
    def ema_model(self) -> ModelEMA:
        return self.ema