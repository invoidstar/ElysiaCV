"""Shared ``forward_train`` / ``forward_test`` for all video models.

Every action-recognition runtime previously copy-pasted its own version.
This module provides the canonical implementations so that changes to
loss computation, accuracy metric, or output format are made **once**.

Usage in a runtime::

    from elysia_cv.models.action_recognition.common_video_forward import (
        forward_train,
        forward_test,
    )

    def _prep(batch, device=None):
        images = batch["inputs"]
        labels = batch["gt_label"]
        if images.dim() == 6:
            images = images.view(-1, *images.shape[2:])
        if device is not None:
            images = images.to(device, non_blocking=True)
            if hasattr(labels, "to"):
                labels = labels.to(device, non_blocking=True)
        return images, labels

    # forward_train(model, batch, prep_fn=_prep)
    # forward_test(model, batch, prep_fn=_prep)
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from elysia_cv.core.registry import LOSSES


def _build_prep_fn(device: torch.device | None) -> Callable:
    """Return a simple prep function that collapses dim==6 and moves to device."""
    def _prep(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["inputs"]
        labels = batch["gt_label"]
        if images.dim() == 6:
            images = images.view(-1, *images.shape[2:])
        if device is not None:
            images = images.to(device, non_blocking=True)
            if hasattr(labels, "to"):
                labels = labels.to(device, non_blocking=True)
        return images, labels
    return _prep


def forward_train(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
    prep_fn: Callable | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard training forward for video classification.

    Args:
        model:   A video model whose ``forward()`` returns ``{"logits": ...}``.
        batch:   A ``PackVideoInputs``-style dict with ``"inputs"`` / ``"gt_label"``.
        device:  Target device (tensors moved if not None).
        prep_fn: Optional ``(batch) -> (images, labels)`` preprocessor.
                 If None, a default dim-6 collapse is used.
        model_kwargs: Extra kwargs passed to ``model(images, **model_kwargs)``.

    Returns:
        ``{"loss": Tensor, "losses": {"ce": Tensor}, "metrics": {"acc": Tensor}}``.
    """
    if prep_fn is None:
        prep_fn = _build_prep_fn(device)

    images, labels = prep_fn(batch)

    if model_kwargs:
        outputs = model(images, **model_kwargs)
    else:
        outputs = model(images)
    logits = outputs["logits"]

    # Handle unbatched outputs
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        labels = labels.unsqueeze(0) if labels.dim() == 0 else labels

    loss_fn = LOSSES.build({"type": "CrossEntropyLoss"})
    loss = loss_fn(logits, labels)

    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean()

    return {
        "loss": loss,
        "losses": {"ce": loss.detach()},
        "metrics": {"acc": acc.detach()},
    }


def forward_test(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device | None = None,
    prep_fn: Callable | None = None,
) -> dict[str, Any]:
    """Standard evaluation forward for video classification.

    Args are identical to :func:`forward_train`.  Returns
    ``{"logits": Tensor, "gt_label": Tensor}`` (no loss).
    """
    if prep_fn is None:
        prep_fn = _build_prep_fn(device)

    images, labels = prep_fn(batch)

    with torch.no_grad():
        outputs = model(images)

    result = {"logits": outputs["logits"].detach()}
    if labels is not None:
        result["gt_label"] = labels
    return result