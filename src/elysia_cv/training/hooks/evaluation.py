"""EvalHook — run evaluation at regular intervals, notify LoggerHook + CheckpointHook."""

from __future__ import annotations

from typing import Any

import torch

from elysia_cv.training.hooks.base import Hook
from elysia_cv.training.distributed import unwrap_model


class EvalHook(Hook):
    """Evaluate model on validation set periodically.

    Uses the METRICS registry to build evaluation metrics, with a
    sensible default (Accuracy + Loss) when none are configured.
    """

    def __init__(
        self,
        dataloader: Any,
        eval_freq: int = 5000,
        metrics: list | None = None,
        save_best_metric: str | None = None,
        ema: Any = None,
        swa: Any = None,
        metric_history: Any = None,
    ) -> None:
        self.dataloader = dataloader
        self.eval_freq = eval_freq

        # Normalize: accept both ["Accuracy"] (legacy) and [{"type":"Accuracy"}]
        if metrics is None:
            self.metrics_cfg = [{"type": "Accuracy"}]
        elif isinstance(metrics[0], str):
            self.metrics_cfg = [{"type": m} for m in metrics]
        else:
            self.metrics_cfg = list(metrics)

        self.save_best_metric = save_best_metric
        self._best_value = -float("inf")
        self._ema = ema  # ModelEMA instance for weight swapping
        self._swa = swa  # SWA instance for weight swapping
        self._metric_history = metric_history  # MetricHistory for curve plotting

    def after_step(self, loop) -> None:
        step = loop.state.global_step
        if step <= 0 or step % self.eval_freq != 0:
            return

        model = loop.model
        raw_model = unwrap_model(model)

        # Swap to EMA/SWA weights for evaluation (if available)
        if self._ema is not None:
            self._ema.apply_shadow(raw_model)
        if self._swa is not None:
            self._swa.apply_shadow(raw_model)

        raw_model.eval()

        try:
            results = self._evaluate(raw_model, loop.device)
        finally:
            # Restore training weights (reverse order)
            if self._swa is not None:
                self._swa.restore(raw_model)
            if self._ema is not None:
                self._ema.restore(raw_model)

        # Primary metric: use save_best_metric if configured, else first numeric result
        primary = 0.0
        if self.save_best_metric and self.save_best_metric in results:
            primary = results[self.save_best_metric]
        elif results:
            primary = next((v for v in results.values() if isinstance(v, (int, float)) and v > 0), 0.0)

        # Record eval results for training-curve plotting
        if self._metric_history is not None:
            self._metric_history.record(step, results)

        raw_model.train()

        improved = primary > self._best_value
        if improved:
            self._best_value = primary

        # Publish to MessageHub (opt-in, best-effort)
        self._write_to_hub(step=step, results=dict(results), primary=primary)

        # Notify LoggerHook (for printing + final summary tracking)
        self._notify_logger(
            loop, "on_eval_result",
            step=step, results=results, primary=primary, improved=improved,
        )

        # Notify CheckpointHook to save best
        if self.save_best_metric and primary > 0:
            for hook in loop.callback._hooks:
                if hasattr(hook, "maybe_save_best"):
                    hook.maybe_save_best(primary, loop)

    def _write_to_hub(self, **kwargs) -> None:
        """Publish eval results to the global MessageHub."""
        try:
            from elysia_cv.training.message_hub import get_message_hub
            hub = get_message_hub()
            step = kwargs.pop("step", 0)
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        hub.update_scalar(f"val/{kk}", float(vv) if isinstance(vv, (int, float)) else 0.)
                elif isinstance(v, (int, float)):
                    hub.update_scalar(f"val/{k}", float(v))
        except Exception:
            pass

    def _notify_logger(self, loop, method: str, **kwargs) -> None:
        """Call *method* on any registered LoggerHook."""
        for hook in loop.callback._hooks:
            if hasattr(hook, method):
                getattr(hook, method)(**kwargs)

    def _evaluate(self, model, device) -> dict[str, float]:
        """Evaluation loop using METRICS registry.

        Auto-detects classification vs detection task at runtime.
        """
        from elysia_cv.core.registry import METRICS, LOSSES

        # Build metric instances from config
        metric_instances: list[Any] = []
        for mc in self.metrics_cfg:
            try:
                metric_instances.append(METRICS.build(mc))
            except Exception:
                pass

        # Detect task type on first batch
        is_detection = False
        first_batch = None
        for batch in self.dataloader:
            first_batch = batch
            break

        if first_batch is not None:
            try:
                outputs = model(first_batch["inputs"].to(device))
                if isinstance(outputs, dict) and "cls_score" in outputs:
                    is_detection = True
            except Exception:
                pass

        if is_detection:
            return self._evaluate_detection(model, device, metric_instances)
        else:
            return self._evaluate_classification(model, device, metric_instances)

    def _evaluate_classification(self, model, device, metric_instances) -> dict[str, float]:
        """Standard classification evaluation."""
        from elysia_cv.core.registry import LOSSES
        loss_fn = LOSSES.build({"type": "CrossEntropyLoss"})

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                if isinstance(batch, dict) and "inputs" in batch:
                    inputs = batch["inputs"].to(device, non_blocking=True)
                    labels = batch.get("gt_label")
                    if labels is not None:
                        labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    logits = outputs.get("logits", outputs) if isinstance(outputs, dict) else outputs

                    if labels is not None:
                        total_loss += loss_fn(logits, labels).item() * labels.size(0)
                        total_samples += labels.size(0)
                        for m in metric_instances:
                            try:
                                m.update(logits, labels)
                            except Exception:
                                pass
                    else:
                        for m in metric_instances:
                            try:
                                m.update(logits)
                            except Exception:
                                pass

        results: dict[str, float] = {}
        if total_samples > 0:
            results["Loss"] = total_loss / total_samples
        for m in metric_instances:
            try:
                val = m.compute()
                if isinstance(val, dict):
                    results.update(val)
                else:
                    results[m.__class__.__name__] = float(val)
            except Exception:
                pass

        if not results or not any(k != "Loss" for k in results):
            results["Accuracy"] = 0.0

        return results

    def _evaluate_detection(self, model, device, metric_instances) -> dict[str, float]:
        """Detection evaluation using forward_test + decode + per-image metric update."""
        from elysia_cv.models.detection.fcos.runtime import forward_test

        with torch.no_grad():
            for batch in self.dataloader:
                outputs = forward_test(model, batch, device=device)
                preds = outputs.get("pred_instances", [])
                gt_bboxes = batch.get("gt_bboxes", [])
                gt_labels = batch.get("gt_labels", [])

                for i in range(len(preds)):
                    for m in metric_instances:
                        try:
                            m.update(
                                pred_boxes=preds[i]["bboxes"].to(device),
                                pred_scores=preds[i]["scores"].to(device),
                                pred_labels=preds[i]["labels"].to(device),
                                gt_boxes=gt_bboxes[i].to(device)
                                if isinstance(gt_bboxes[i], torch.Tensor) else gt_bboxes[i],
                                gt_labels=gt_labels[i].to(device)
                                if isinstance(gt_labels[i], torch.Tensor) else gt_labels[i],
                            )
                        except Exception:
                            pass

        results: dict[str, float] = {}
        for m in metric_instances:
            try:
                val = m.compute()
                if isinstance(val, dict):
                    results.update(val)
                else:
                    results[m.__class__.__name__] = float(val)
            except Exception:
                pass

        if not results:
            results["mAP"] = 0.0

        return results