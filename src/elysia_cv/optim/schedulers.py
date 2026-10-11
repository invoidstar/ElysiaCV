"""LR scheduler builders — from config dicts to torch.optim.lr_scheduler instances."""

from __future__ import annotations

from typing import Any

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build an LR scheduler from a config dict.

    Config keys:
      - ``type``: scheduler name (CosineAnnealing, MultiStepLR, ...)
      - ``steps``: total training steps
      - ``base_lr``: initial learning rate
      - ``min_lr``: minimum LR (for cosine)
      - ``warmup_steps``: linear warmup steps
      - ``params``: extra kwargs passed to scheduler constructor
    """
    sched_type = cfg.get("type", "CosineAnnealing")
    steps = int(cfg.get("steps", 100000))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    min_lr = float(cfg.get("min_lr", 0.0))
    base_lr = float(cfg.get("base_lr", 0.001))
    params = dict(cfg.get("params", {}))

    # Build the main scheduler
    if sched_type == "MultiStepLR":
        milestones = params.get("milestones", [int(steps * 0.6), int(steps * 0.8)])
        gamma = params.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif sched_type == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=min_lr)
    elif sched_type == "StepLR":
        step_size = params.get("step_size", 30)
        gamma = params.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=params.get("mode", "min"),
            factor=params.get("factor", 0.1), patience=params.get("patience", 10))
    elif sched_type == "ConstantLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"unknown scheduler type: {sched_type!r}")

    # Wrap with warmup if needed
    if warmup_steps > 0:
        scheduler = _WarmupScheduler(scheduler, warmup_steps=warmup_steps, base_lr=base_lr)

    return scheduler


class _WarmupScheduler:
    """Linear warmup wrapper around a base scheduler."""

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        warmup_steps: int,
        base_lr: float,
    ) -> None:
        self._scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self._step = 0

    def step(self) -> None:
        self._step += 1
        if self._step <= self.warmup_steps:
            # Linear warmup: lr = base_lr * (step / warmup_steps)
            scale = self._step / max(self.warmup_steps, 1)
            lr = self.base_lr * scale
            for param_group in self._scheduler.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            self._scheduler.step()

    def state_dict(self) -> dict:
        return {"warmup_step": self._step, "scheduler": self._scheduler.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["warmup_step"]
        self._scheduler.load_state_dict(state["scheduler"])

    def get_last_lr(self) -> list[float]:
        return self._scheduler.get_last_lr()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduler, name)