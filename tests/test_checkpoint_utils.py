"""Tests for checkpoint_utils.py."""

from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn

from elysia_cv.checkpoint_utils import find_checkpoint, load_checkpoint


def test_find_checkpoint_best_accuracy():
    with tempfile.TemporaryDirectory() as tmp:
        # Create checkpoints with different mtimes
        for name in ["checkpoint_step_100.pt", "checkpoint_best.pt",
                     "checkpoint_best_Accuracy.pt"]:
            path = os.path.join(tmp, name)
            torch.save({"model_state_dict": {}}, path)

        # prefer_best should pick best_Accuracy.pt
        found = find_checkpoint(tmp, prefer_best=True)
        assert found is not None
        assert "best_Accuracy" in os.path.basename(found)


def test_find_checkpoint_final():
    with tempfile.TemporaryDirectory() as tmp:
        torch.save({"model": {}}, os.path.join(tmp, "checkpoint_final.pt"))
        found = find_checkpoint(tmp, prefer_final=True)
        assert found is not None
        assert "final" in os.path.basename(found)


def test_find_checkpoint_empty():
    with tempfile.TemporaryDirectory() as tmp:
        assert find_checkpoint(tmp) is None


def test_load_checkpoint_raw_state_dict():
    model = nn.Linear(10, 2)
    state = model.state_dict()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(state, f.name)  # raw state_dict, no container wrapper
    try:
        model2 = nn.Linear(10, 2)
        load_checkpoint(f.name, model2, strict=True)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    finally:
        os.unlink(f.name)


def test_load_checkpoint_strips_module_prefix():
    """DDP wrapped models have 'module.' prefix — must be stripped."""
    model = nn.Linear(10, 2)
    state = {f"module.{k}": v for k, v in model.state_dict().items()}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({"model_state_dict": state}, f.name)
    try:
        model2 = nn.Linear(10, 2)
        load_checkpoint(f.name, model2, strict=False)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    finally:
        os.unlink(f.name)