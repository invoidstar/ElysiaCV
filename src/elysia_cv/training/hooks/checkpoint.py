"""CheckpointHook — save model / optimizer / scheduler state with TTY-aware output."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from elysia_cv.training.hooks.base import Hook
from elysia_cv.training.distributed import unwrap_model


class CheckpointHook(Hook):
    """Save checkpoints at regular intervals and on completion.

    Output is TTY-aware: in interactive terminals the message appears
    above the tqdm progress bar; in piped/tee mode it prints as a line.
    """

    def __init__(
        self,
        output_dir: str,
        save_freq: int = 5000,
        save_best_metric: str | None = None,
        save_final: bool = True,
        max_keep: int = 5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_freq = save_freq
        self.save_best_metric = save_best_metric
        self.save_final = save_final
        self.max_keep = max_keep
        self.best_value = -float("inf")
        self._saved_paths: list[Path] = []
        self._is_tty = sys.stdout.isatty()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def after_step(self, loop) -> None:
        step = loop.state.global_step
        if step > 0 and step % self.save_freq == 0:
            self._save(step, loop, tag=f"step_{step}")

    def after_epoch(self, loop) -> None:
        pass  # Could save epoch checkpoints too

    def after_run(self, loop) -> None:
        if self.save_final:
            self._save(loop.state.global_step, loop, tag="final")
        self._cleanup_old()

    def maybe_save_best(self, metric_value: float, loop) -> None:
        """Call from EvalHook to save best checkpoint."""
        if self.save_best_metric is None:
            return
        if metric_value > self.best_value:
            self.best_value = metric_value
            self._save(loop.state.global_step, loop, tag=f"best_{self.save_best_metric}")

    def _save(self, step: int, loop, tag: str) -> None:
        model = unwrap_model(loop.model)
        checkpoint = {
            "step": step,
            "epoch": loop.state.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": loop.optimizer.state_dict(),
        }
        if hasattr(loop, "scheduler") and loop.scheduler is not None:
            checkpoint["scheduler_state_dict"] = loop.scheduler.state_dict()

        filename = f"checkpoint_{tag}.pt"
        path = self.output_dir / filename
        torch.save(checkpoint, str(path))
        self._saved_paths.append(path)

        # Write step metadata
        meta = {"step": step, "tag": tag, "path": str(path)}
        (self.output_dir / f"checkpoint_{tag}.json").write_text(json.dumps(meta, indent=2))

        msg = f"  [checkpoint] {filename}"

        if self._is_tty:
            import tqdm
            tqdm.tqdm.write(msg)
        else:
            print(msg, flush=True)

    def _cleanup_old(self) -> None:
        if len(self._saved_paths) <= self.max_keep:
            return
        for path in self._saved_paths[:-self.max_keep]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._saved_paths = self._saved_paths[-self.max_keep:]

    def load_checkpoint(self, path: str, loop) -> int:
        """Restore training state from checkpoint; return the step to resume from."""
        ckpt = torch.load(path, map_location="cpu")
        model = unwrap_model(loop.model)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            loop.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and hasattr(loop, "scheduler"):
            loop.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return int(ckpt.get("step", 0))