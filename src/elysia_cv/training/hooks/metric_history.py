"""MetricHistory — track metrics over time and plot training curves.

Registers as a training hook that records evaluation results and
automatically saves a training-curves PNG at the end of training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from elysia_cv.training.hooks.base import Hook


class MetricHistory(Hook):
    """Record evaluation metrics and plot training curves.

    Hooks into ``after_step`` to capture eval results and ``after_run`` to
    save the plot as ``analysis/training_curves.png`` in the output directory.

    Usage in YAML::

        eval:
          metric_history: true          # enables tracking
    """

    def __init__(self, output_dir: str = "./outputs") -> None:
        self.output_dir = output_dir
        self._history: dict[str, list[float]] = {}
        self._steps: list[int] = []

    # ── Hook API ──────────────────────────────────────────────

    def after_step(self, loop: Any) -> None:
        """Capture the latest eval results if present."""
        metrics = getattr(loop.state, "metrics", None)
        if not metrics or not isinstance(metrics, dict):
            return

        step = loop.state.global_step
        # Avoid duplicate records for the same step
        if self._steps and self._steps[-1] == step:
            return

        self._steps.append(step)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self._history.setdefault(key, []).append(val)

    def record(self, step: int, results: dict[str, float]) -> None:
        """Called by EvalHook to record evaluation results."""
        if step in self._steps:
            return  # already recorded this step
        self._steps.append(step)
        for key, val in results.items():
            if isinstance(val, (int, float)):
                self._history.setdefault(key, []).append(val)

    def after_run(self, loop: Any) -> None:
        """Plot curves and save to disk."""
        if not self._history or not self._steps:
            return

        try:
            self._plot_and_save()
        except Exception:
            pass  # Non-critical; never crash training

    # ── Plotting ──────────────────────────────────────────────

    def _plot_and_save(self) -> None:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        n_metrics = len(self._history)
        if n_metrics == 0:
            return

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, self._history.items()):
            ax.plot(self._steps[: len(values)], values, marker="o", markersize=3, linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel(name.replace("_", " ").title())
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Training Curves", fontsize=13, y=1.02)
        plt.tight_layout()

        out_path = Path(self.output_dir) / "analysis"
        out_path.mkdir(parents=True, exist_ok=True)
        file = out_path / "training_curves.png"
        plt.savefig(str(file), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── API for external consumers ────────────────────────────

    def get_history(self) -> dict[str, list[float]]:
        """Return the full recorded history."""
        return dict(self._history)

    @property
    def steps(self) -> list[int]:
        return list(self._steps)