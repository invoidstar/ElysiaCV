"""TrainerLoop — the main training loop, policy-agnostic.

Adapted from ElysiaRobot's shared_trainer_loop.py. The loop itself knows
nothing about classification vs detection vs segmentation; all task-specific
logic lives in the model's ``runtime`` and in hooks.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from elysia_cv.training.loop_state import TrainerLoopContext, TrainerLoopState
from elysia_cv.training.distributed import DistributedContext, gradient_sync_context
from elysia_cv.training.strategy import ExecutionStrategy
from elysia_cv.training.cadence import CadenceController
from elysia_cv.training.hooks.base import HookBus
from elysia_cv.training.metrics_runtime import accumulate_micro_metrics, finalize_micro_metrics


class TrainerLoop:
    """Device- and task-agnostic training loop."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
            dataloader: Any,
        device: torch.device,
        max_steps: int,
        strategy: ExecutionStrategy,
        callback: HookBus,
        scheduler: Any = None,
        distributed: DistributedContext | None = None,
        cadence: CadenceController | None = None,
        grad_clip_norm: float | None = None,
        autocast_ctx: Any = None,
        grad_scaler: Any = None,
        batch_transforms: list | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.max_steps = max_steps
        self.strategy = strategy
        self.callback = callback
        self.scheduler = scheduler
        self.distributed = distributed or DistributedContext(False, 0, 0, 1)
        self.cadence = cadence or CadenceController()
        self.state = TrainerLoopState()
        self._data_iter = iter(dataloader)
        self._autocast_ctx = autocast_ctx  # torch.amp.autocast or nullcontext
        self._grad_scaler = grad_scaler    # torch.cuda.amp.GradScaler or None
        self._batch_transforms = batch_transforms or []

    def run(self) -> None:
        """Execute the full training loop."""
        self.callback.before_run(self)
        self.state.should_stop = False

        while self.state.global_step < self.max_steps and not self.state.should_stop:
            self.state.epoch += 1

            # Correctly seed the DistributedSampler so each epoch shuffles differently
            sampler = getattr(self.dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.state.epoch)

            self.callback.before_epoch(self)

            self._run_epoch()

            self.callback.after_epoch(self)

        self.callback.after_run(self)

    def _run_epoch(self) -> None:
        while self.state.global_step < self.max_steps and not self.state.should_stop:
            self._run_optimizer_step()

    def _run_optimizer_step(self) -> None:
        """Run one optimizer step, possibly spanning multiple micro-batches."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        micro_accumulator: dict[str, dict[str, Any]] = {}
        total_loss = 0.0
        micro_count = 0

        for micro_idx in range(self.strategy.grad_accum_steps):
            # Fetch batch
            try:
                batch = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(self.dataloader)
                batch = next(self._data_iter)

            # Apply batch-level transforms (MixUp etc.)
            for bt in self._batch_transforms:
                batch = bt(batch)

            # Determine sync
            is_sync = self.strategy.is_sync_step(micro_idx)
            self.cadence.set_step(self.state.micro_step)
            self.callback.before_micro_step(self)

            with gradient_sync_context(self.model, self.distributed, is_sync):
                outputs = self._forward(batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                scaled_loss = self.strategy.scale_loss(loss)
                if self._grad_scaler is not None:
                    scaled_loss = self._grad_scaler.scale(scaled_loss)
                scaled_loss.backward()

            micro_count += 1
            total_loss += loss.detach().item() if hasattr(loss, "detach") else float(loss)
            if isinstance(outputs, dict):
                accumulate_micro_metrics(micro_accumulator, outputs.get("metrics", {}))
                accumulate_micro_metrics(micro_accumulator, outputs.get("losses", {}))

            self.state.micro_step += 1
            self.callback.after_micro_step(self)

            # Sync point: optimizer step
            if is_sync:
                if self._grad_scaler is not None:
                    self._grad_scaler.unscale_(self.optimizer)
                self.strategy.clip_gradients(self.model)
                if self._grad_scaler is not None:
                    self._grad_scaler.step(self.optimizer)
                    self._grad_scaler.update()
                else:
                    self.optimizer.step()

                self.state.global_step += 1
                self.state.last_loss = total_loss / max(micro_count, 1)

                # Finalize micro-batch metrics
                finalized = finalize_micro_metrics(micro_accumulator)
                self.state.metrics = finalized

                self.callback.after_step(self)
                break  # Ready for next optimizer step

    def _forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run one forward pass. Override in subclasses for custom logic.

        YOLO / Video models use ``forward_train(batch)`` which takes the full
        batch dict.  Other models follow the ``model(inputs, **labels)`` pattern.
        """
        _KNOWN_LABEL_KEYS = {"gt_label", "gt_label_soft", "gt_bboxes", "gt_semantic_seg", "gt_masks", "labels"}

        autocast = self._autocast_ctx if self._autocast_ctx is not None else _nullcontext()

        with autocast:
            if isinstance(batch, dict) and "inputs" in batch:
                # YOLO / Video: batch dict consumed internally by forward_train
                if getattr(self.model, "_is_yolo", False) or getattr(self.model, "_is_video", False):
                    batch_on_device = {
                        k: v.to(self.device, non_blocking=True) if hasattr(v, "to") else v
                        for k, v in batch.items()
                    }
                    return self.model._forward_train_fn(self.model, batch_on_device)

                # Standard: unpack inputs + label kwargs
                inputs = batch["inputs"].to(self.device, non_blocking=True)
                labels = {
                    k: v.to(self.device, non_blocking=True) if hasattr(v, "to") else v
                    for k, v in batch.items()
                    if k in _KNOWN_LABEL_KEYS
                }
                return self.model(inputs, **labels)
            return self.model(batch)


class _nullcontext:
    """Trivial context manager (Python 3.7+ has contextlib.nullcontext)."""
    def __enter__(self) -> None: return None
    def __exit__(self, *args: Any) -> None: pass