"""Shared checkpoint discovery and loading utilities.

Previously, checkpoint auto-discovery was copy-pasted three times in
``cli.py`` (:func:`_resolve_resume_path`, ``_cmd_test``, ``_cmd_infer`).
This module provides the canonical versions so that checkpoint search
logic is defined once.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import torch
import torch.nn as nn


def find_checkpoint(
    output_dir: str,
    *,
    prefer_best: bool = True,
    prefer_final: bool = False,
) -> str | None:
    """Discover the most suitable checkpoint in *output_dir*.

    When *prefer_best* is True (default, used by test/infer)::

        best_Accuracy.pt → best.pt → final.pt

    When *prefer_final* is True (used by resume)::

        final.pt → best_*.pt → step_*.pt

    Args:
        output_dir:   Checkpoint directory.
        prefer_best:  Search for ``checkpoint_best_*`` first.
        prefer_final: Search for ``checkpoint_final.pt`` first.

    Returns:
        Absolute path string, or ``None`` if nothing matched.
    """
    root = str(output_dir)

    if prefer_final:
        candidates = [
            os.path.join(root, "checkpoint_final.pt"),
            os.path.join(root, "checkpoint_best_*.pt"),
            os.path.join(root, "checkpoint_step_*.pt"),
        ]
    elif prefer_best:
        candidates = [
            os.path.join(root, "checkpoint_best_Accuracy.pt"),
            os.path.join(root, "checkpoint_best.pt"),
            os.path.join(root, "checkpoint_final.pt"),
        ]
    else:
        candidates = [
            os.path.join(root, "checkpoint_best_Accuracy.pt"),
            os.path.join(root, "checkpoint_best.pt"),
            os.path.join(root, "checkpoint_final.pt"),
            os.path.join(root, "checkpoint_best_*.pt"),
            os.path.join(root, "checkpoint_step_*.pt"),
        ]

    for pattern in candidates:
        if "*" in pattern:
            matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if matches:
                return matches[0]
        elif os.path.isfile(pattern):
            return pattern

    return None


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    device: torch.device | str | None = None,
    *,
    strict: bool = False,
) -> nn.Module:
    """Load a checkpoint into *model*.

    Handles both raw ``state_dict`` and checkpoint containers (keys
    ``model_state_dict`` or ``model``).  Strips ``"module."`` prefixes
    from DDP-wrapped models.
    """
    ckpt = torch.load(ckpt_path, map_location=device or "cpu")

    # Unwrap container
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # raw state_dict

    # Strip DDP wrapper prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=strict)
    return model