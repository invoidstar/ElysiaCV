"""Execution strategy — gradient accumulation with DDP no_sync.

The core pattern adapted from ElysiaRobot state_machine.py:
- Each "optimizer step" can span N micro-batches.
- Only the LAST micro-batch triggers gradient sync + optimizer step.
- All previous micro-batches use model.no_sync() to skip all-reduce.
"""

from __future__ import annotations

from typing import Any


class ExecutionStrategy:
    """Handles gradient accumulation and DDP sync decisions."""

    def __init__(self, *, grad_accum_steps: int = 1, grad_clip_norm: float | None = None) -> None:
        self.grad_accum_steps = max(grad_accum_steps, 1)
        self.grad_clip_norm = grad_clip_norm

    def is_sync_step(self, micro_idx: int) -> bool:
        """Return True if this micro-step should trigger gradient sync."""
        return micro_idx == self.grad_accum_steps - 1

    def scale_loss(self, loss: Any) -> Any:
        """Scale loss for gradient accumulation: loss / accum_steps."""
        return loss / self.grad_accum_steps

    def clip_gradients(self, model: Any) -> None:
        """Clip gradients if configured."""
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            import torch
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)