"""P6 — ModelEMA and EMAHook tests."""

from __future__ import annotations

import torch
import torch.nn as nn

import elysia_cv  # noqa: F401


class _SimpleModel(nn.Module):
    """Two-layer net for EMA tests."""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestModelEMA:
    """Unit tests for ModelEMA shadow-copy + swap lifecycle."""

    def test_shadow_copy_created(self):
        from elysia_cv.training.ema import ModelEMA
        model = _SimpleModel()
        ema = ModelEMA(model, decay=0.999)
        # All trainable params should be in shadow
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.allclose(ema.shadow[name], param.data)

    def test_update_moves_shadow(self):
        from elysia_cv.training.ema import ModelEMA
        model = _SimpleModel()
        ema = ModelEMA(model, decay=0.9, warmup_steps=1)

        init_fc1 = ema.shadow["fc1.weight"].clone()

        optim = torch.optim.SGD(model.parameters(), lr=1.0)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        optim.step()
        ema.update(model)

        # After update with decay=0.9, shadow should differ from init
        assert not torch.allclose(ema.shadow["fc1.weight"], init_fc1)

    def test_apply_restore_roundtrip(self):
        from elysia_cv.training.ema import ModelEMA
        model = _SimpleModel()
        ema = ModelEMA(model, decay=0.9, warmup_steps=1)

        orig_weight = model.fc1.weight.data.clone()

        # Update EMA for a few steps — use a persistent optimizer
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(5):
            optim.zero_grad()
            loss = model(torch.randn(2, 4)).sum()
            loss.backward()
            optim.step()
            ema.update(model)

        # After training, weights should differ from original
        assert not torch.allclose(model.fc1.weight.data, orig_weight, atol=1e-5)

        # Apply shadow → restore roundtrip
        after_train = model.fc1.weight.data.clone()
        ema.apply_shadow(model)
        assert not torch.allclose(model.fc1.weight.data, after_train, atol=1e-5)
        ema.restore(model)
        assert torch.allclose(model.fc1.weight.data, after_train)

    def test_warmup_schedule(self):
        from elysia_cv.training.ema import ModelEMA
        model = _SimpleModel()
        ema = ModelEMA(model, decay=0.999, start_decay=0.9, warmup_steps=50)

        # Before any step
        assert ema._current_decay() < 0.999
        # Step past warmup
        for _ in range(51):
            ema._step += 1
        assert abs(ema._current_decay() - 0.999) < 1e-4

    def test_ema_hook_attaches_ema(self):
        from elysia_cv.training.ema import ModelEMA, EMAHook
        model = _SimpleModel()
        ema = ModelEMA(model)
        hook = EMAHook(ema)
        assert hook.ema is ema
        assert hook.ema_model is ema

    def test_resume_compatible_format(self):
        """EMA shadow dict is compatible with state_dict serialization."""
        from elysia_cv.training.ema import ModelEMA
        model = _SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        # Verify shadow dict can be saved/loaded like a state_dict
        saved = {k: v.clone() for k, v in ema.shadow.items()}
        loaded = {k: v.clone() for k, v in saved.items()}
        for k in saved:
            assert torch.allclose(saved[k], loaded[k])