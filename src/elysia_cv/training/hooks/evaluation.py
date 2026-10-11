"""EvalHook — run evaluation at regular intervals."""

from __future__ import annotations

from typing import Any

import torch

from elysia_cv.training.hooks.base import Hook
from elysia_cv.training.distributed import unwrap_model


class EvalHook(Hook):
    """Evaluate model on validation set periodically."""

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

    def after_step(self, loop) -> None:
        step = loop.state.global_step
        if step <= 0 or step % self.eval_freq != 0:
            return

        model = loop.model
        model.eval()
        results = self._evaluate(model, loop.device)

        if loop.distributed.enabled and loop.distributed.is_main_process:
            print(f"\n[Eval @ step {step}] " + "  ".join(
                f"{k}: {v:.4f}" for k, v in results.items() if isinstance(v, float)
            ))

        model.train()

        # Notify CheckpointHook about best metric
        if self.save_best_metric and self.save_best_metric in results:
            for hook in loop.callbacks._hooks:
                if hasattr(hook, "maybe_save_best"):
                    hook.maybe_save_best(results[self.save_best_metric], loop)

    def _evaluate(self, model, device) -> dict[str, float]:
        """Basic evaluation loop — override for custom logic."""
        total = 0
        correct = 0

        with torch.no_grad():
            for batch in self.dataloader:
                if isinstance(batch, dict) and "inputs" in batch:
                    inputs = batch["inputs"].to(device, non_blocking=True)
                    labels = batch.get("gt_label")
                    if labels is not None:
                        labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs.get("logits", outputs.get("output"))
                    if labels is not None:
                        pred = outputs.argmax(dim=1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

        acc = correct / max(total, 1) * 100
        return {"Accuracy": acc}