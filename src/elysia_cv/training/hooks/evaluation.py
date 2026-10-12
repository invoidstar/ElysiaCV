"""EvalHook — run evaluation at regular intervals, notify LoggerHook + CheckpointHook."""

from __future__ import annotations

from typing import Any

import torch

from elysia_cv.training.hooks.base import Hook
from elysia_cv.training.distributed import unwrap_model


class EvalHook(Hook):
    """Evaluate model on validation set periodically.

    Informs LoggerHook (for logging + final summary) and CheckpointHook
    (for best-checkpoint saving) via the hook bus.
    """

    def __init__(
        self,
        dataloader: Any,
        eval_freq: int = 5000,
        metrics: list[str] | None = None,
        save_best_metric: str | None = None,
    ) -> None:
        self.dataloader = dataloader
        self.eval_freq = eval_freq
        self.metric_names = metrics or ["Accuracy"]
        self.save_best_metric = save_best_metric
        self._best_value = -float("inf")

    def after_step(self, loop) -> None:
        step = loop.state.global_step
        if step <= 0 or step % self.eval_freq != 0:
            return

        model = loop.model
        raw_model = unwrap_model(model)
        raw_model.eval()

        results = self._evaluate(raw_model, loop.device)
        acc = results.get("Accuracy", 0.0)
        val_loss = results.get("Loss", 0.0)

        raw_model.train()

        improved = acc > self._best_value
        if improved:
            self._best_value = acc

        # Notify LoggerHook (for printing + final summary tracking)
        self._notify_logger(loop, "on_eval_result", step=step, acc=acc, val_loss=val_loss, improved=improved)

        # Notify CheckpointHook to save best
        if self.save_best_metric and acc > 0:
            for hook in loop.callback._hooks:
                if hasattr(hook, "maybe_save_best"):
                    hook.maybe_save_best(acc, loop)

    def _notify_logger(self, loop, method: str, **kwargs) -> None:
        """Call *method* on any registered LoggerHook."""
        for hook in loop.callback._hooks:
            if hasattr(hook, method):
                getattr(hook, method)(**kwargs)

    def _evaluate(self, model, device) -> dict[str, float]:
        """Basic evaluation loop."""
        total = 0
        correct = 0
        total_loss = 0.0

        # Use FocalLoss for cassava classification
        from elysia_cv.core.registry import LOSSES
        loss_fn = LOSSES.build({"type": "FocalLoss", "alpha": 0.25, "gamma": 2.0})

        with torch.no_grad():
            for batch in self.dataloader:
                if isinstance(batch, dict) and "inputs" in batch:
                    inputs = batch["inputs"].to(device, non_blocking=True)
                    labels = batch.get("gt_label")
                    if labels is not None:
                        labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        logits = outputs.get("logits", outputs)
                    else:
                        logits = outputs
                    if labels is not None:
                        total_loss += loss_fn(logits, labels).item() * labels.size(0)
                        pred = logits.argmax(dim=1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

        total = max(total, 1)
        return {
            "Accuracy": correct / total * 100,
            "Loss": total_loss / total,
        }