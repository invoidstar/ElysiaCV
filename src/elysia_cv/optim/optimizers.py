"""Optimizer builders — from config dicts to torch.optim instances.

Adapted from ElysiaRobot's optimizer_service.py, simplified for CV tasks.
Supports config-driven construction via ``OPTIMIZERS`` registry.
"""

from __future__ import annotations

from typing import Any

import torch

from elysia_cv.core.registry import OPTIMIZERS


def build_optimizer(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    params_override: list[dict[str, Any]] | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer from a config dict.

    Uses the ``OPTIMIZERS`` registry. Config keys:
      - ``type``: registered optimizer name (SGD, AdamW, ...)
      - ``params``: kwargs forwarded to constructor
      - ``paramwise_cfg``: optional per-parameter-group overrides
    """
    # Collect parameters with optional per-group config
    if params_override is not None:
        param_groups = params_override
    else:
        paramwise_cfg = cfg.get("paramwise_cfg", {})
        param_groups = _build_param_groups(model, paramwise_cfg)

    optim_type = cfg.get("type", "AdamW")
    optim_params = dict(cfg.get("params", {}))

    if isinstance(param_groups, list) and len(param_groups) > 0:
        return OPTIMIZERS.build({"type": optim_type, "params": param_groups, **optim_params})
    else:
        return OPTIMIZERS.build({"type": optim_type, "params": model.parameters(), **optim_params})


def _build_param_groups(
    model: torch.nn.Module,
    paramwise_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build per-parameter groups with custom LR/weight_decay overrides."""
    groups: list[dict[str, Any]] = []

    # Default: all params
    default_params = [p for p in model.parameters() if p.requires_grad]
    if not default_params:
        return groups

    # weight_decay split (no_decay for biases, norms)
    no_decay_names = paramwise_cfg.get("no_decay_names", [])
    if not no_decay_names:
        no_decay_names = ["bias", "norm", "bn", "ln", "gn"]

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    weight_decay = paramwise_cfg.get("weight_decay", 0.01)
    groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return groups


# ---------------------------------------------------------------------------
# Built-in optimizer registrations
# ---------------------------------------------------------------------------


@OPTIMIZERS.register_module()
def SGD(params: Any, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0,
        nesterov: bool = False, **kwargs: Any) -> torch.optim.Optimizer:
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                           nesterov=nesterov, **kwargs)


@OPTIMIZERS.register_module()
def AdamW(params: Any, lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999),
          eps: float = 1e-8, weight_decay: float = 0.01, **kwargs: Any) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)