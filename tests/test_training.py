"""End-to-end training tests — full pipeline from config to checkpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

import elysia_cv  # triggers registry population


class TestFullPipeline:
    """End-to-end tests: build model → forward → train → checkpoint."""

    def test_model_build_and_forward(self):
        """Verify the full backbone + head + loss + metric pipeline."""
        from elysia_cv.core.registry import BACKBONES, HEADS, LOSSES, METRICS

        backbone = BACKBONES.build({"type": "ResNet", "depth": 18})
        head = HEADS.build({"type": "LinearClsHead", "num_classes": 10, "in_channels": 512})
        x = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 10, (4,))

        features = backbone(x)
        logits = head(features)
        loss = LOSSES.build({"type": "CrossEntropyLoss"})(logits, labels)
        metric = METRICS.build({"type": "Accuracy"})
        metric.update(logits, labels)

        assert features.shape == (4, 512)
        assert logits.shape == (4, 10)
        assert loss.item() > 0
        assert 0.0 <= metric.compute() <= 1.0

    def test_runtime_build_and_train(self):
        """Verify build_model + forward_train from the runtime."""
        from elysia_cv.models.classification.resnet.runtime import (
            build_model,
            forward_train,
        )
        cfg = {
            "model": {
                "backbone": {"type": "ResNet", "depth": 18},
                "head": {"type": "LinearClsHead", "num_classes": 10, "in_channels": 512},
            }
        }
        model = build_model(cfg)
        x = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 10, (4,))
        result = forward_train(model, {"inputs": x, "gt_label": labels})

        assert "loss" in result
        assert "losses" in result
        assert "metrics" in result
        assert result["loss"].item() > 0

    def test_optimizer_and_scheduler(self):
        """Build optimizer and scheduler from config."""
        import torch.nn as nn
        from elysia_cv.optim.optimizers import build_optimizer
        from elysia_cv.optim.schedulers import build_scheduler

        m = nn.Linear(10, 2)
        opt = build_optimizer(m, {"type": "SGD", "params": {"lr": 0.01, "momentum": 0.9}})
        assert isinstance(opt, torch.optim.SGD)
        sched = build_scheduler(opt, {"type": "CosineAnnealing", "steps": 100, "base_lr": 0.01, "warmup_steps": 10})
        sched.step()
        assert sched.get_last_lr()[0] > 0

    def test_training_loop_single_step(self):
        """Run TrainerLoop for a few steps on fake data."""
        from elysia_cv.models.classification.resnet.runtime import build_model
        from elysia_cv.training.loop import TrainerLoop
        from elysia_cv.training.strategy import ExecutionStrategy
        from elysia_cv.training.hooks.base import HookBus
        from elysia_cv.training.hooks.logger import LoggerHook
        from elysia_cv.optim.optimizers import build_optimizer
        from elysia_cv.optim.schedulers import build_scheduler

        # Fake dataset
        class FakeDataset:
            def __init__(self, n=32):
                self.n = n
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                return {
                    "inputs": torch.randn(3, 224, 224),
                    "gt_label": torch.randint(0, 10, ()),
                }

        from torch.utils.data import DataLoader
        loader = DataLoader(FakeDataset(), batch_size=4, shuffle=True)

        # Build model
        model = build_model({
            "model": {
                "backbone": {"type": "ResNet", "depth": 18},
                "head": {"type": "LinearClsHead", "num_classes": 10, "in_channels": 512},
            }
        })

        # Build optimizer + scheduler
        optimizer = build_optimizer(model, {"type": "SGD", "params": {"lr": 0.01, "momentum": 0.9}})
        scheduler = build_scheduler(optimizer, {"type": "CosineAnnealing", "steps": 100, "base_lr": 0.01})

        # Strategy
        strategy = ExecutionStrategy(grad_accum_steps=1)

        # Hooks
        hooks = HookBus()
        hooks.register(LoggerHook(log_freq=10))

        # Loop
        loop = TrainerLoop(
            model=model,
            optimizer=optimizer,
            dataloader=loader,
            device=torch.device("cpu"),  # CPU for test reliability
            max_steps=10,
            strategy=strategy,
            callback=hooks,
            scheduler=scheduler,
        )

        # Need to override _forward for our runtime
        from elysia_cv.models.classification.resnet.runtime import forward_train
        def my_forward(batch):
            return forward_train(model, batch, device=torch.device("cpu"))

        loop._forward = lambda batch: my_forward(batch)

        loop.run()

        assert loop.state.global_step == 10
        assert loop.state.last_loss > 0

    def test_checkpoint_save_and_load(self):
        """Save and restore a checkpoint."""
        from elysia_cv.training.hooks.checkpoint import CheckpointHook

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "ckpt"
            hook = CheckpointHook(
                output_dir=str(output_dir),
                save_freq=5,
                save_final=True,
                max_keep=3,
            )

            # Create a minimal model
            import torch.nn as nn

            class MockLoop:
                pass

            mock_loop = MockLoop()
            mock_loop.model = nn.Linear(10, 2)
            mock_loop.optimizer = torch.optim.SGD(mock_loop.model.parameters(), lr=0.01)
            mock_loop.device = torch.device("cpu")
            mock_loop.callbacks = None

            class _State:
                global_step = 0
                epoch = 0
                last_loss = None
            mock_loop.state = _State()

            class _Dist:
                enabled = False
                is_main_process = True
                local_rank = 0
                rank = 0
                world_size = 1
            mock_loop.distributed = _Dist()

            # Simulate training
            for step in range(1, 11):
                mock_loop.state.global_step = step
                x = torch.randn(4, 10)
                y = torch.randn(4, 2)
                loss = ((mock_loop.model(x) - y) ** 2).mean()
                loss.backward()
                mock_loop.optimizer.step()
                mock_loop.optimizer.zero_grad()
                hook.after_step(mock_loop)

            hook.after_run(mock_loop)

            # Verify files exist
            ckpt_files = list(output_dir.glob("checkpoint_*.pt"))
            assert len(ckpt_files) >= 2, f"Expected >=2 checkpoints, got {len(ckpt_files)}"

            # Load the final checkpoint
            final_path = output_dir / "checkpoint_final.pt"
            assert final_path.exists()
            loaded_step = hook.load_checkpoint(str(final_path), mock_loop)
            assert loaded_step == 10

    def test_plugin_discovery_includes_resnet(self):
        """Verify ResNet plugin is auto-discovered."""
        from elysia_cv.models.registry import ModelRegistry

        reg = ModelRegistry.discover()
        assert "resnet" in reg

        plugin = reg.get("resnet")
        assert plugin.plugin.task_type == "classification"
        specs = list(plugin.plugin.model_specs())
        assert len(specs) == 4  # 18, 34, 50, 101
        names = [s.name for s in specs]
        assert "resnet18" in names
        assert "resnet50" in names

    def test_training_plan_compiles(self):
        """TrainingPlan compiles from config + validates."""
        from elysia_cv.config import load_cv_config_with_object
        from elysia_cv.training.plan import TrainingPlan

        cfg_path = Path(__file__).parent.parent / "configs" / "classification" / "resnet50_imagenet.yaml"
        if not cfg_path.exists():
            from pytest import skip
            skip("config file not found")

        cfg = load_cv_config_with_object(str(cfg_path))
        plan = TrainingPlan.from_config(cfg)
        assert plan.run["stage"] == "train"
        assert plan.model["type"] == "resnet50"

    def test_hook_bus_lifecycle(self):
        """Verify HookBus dispatches lifecycle events."""
        from elysia_cv.training.hooks.base import Hook, HookBus

        events = []

        class LifecycleHook(Hook):
            def before_run(self, loop):
                events.append("before_run")
            def after_step(self, loop):
                events.append(f"after_step_{loop.state.global_step if loop else 0}")

        bus = HookBus()
        bus.register(LifecycleHook())

        class MockLoop:
            class State:
                global_step = 5
                epoch = 0
                last_loss = 1.0
                metrics = {}
                micro_step = 0
                should_stop = False
            state = State()

        m = MockLoop()
        bus.before_run(m)
        assert events == ["before_run"]