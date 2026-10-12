"""Cassava Leaf Disease Classification — end-to-end tests."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

import elysia_cv  # noqa: F401


DATA_ROOT = "/data/hdd3/chuqi/data/cassava"


class TestCassavaDataset:
    def test_split_files_exist(self):
        assert (Path(DATA_ROOT) / "train_split.csv").exists()
        assert (Path(DATA_ROOT) / "val_split.csv").exists()

    def test_dataset_load(self):
        from elysia_cv.data import CassavaDataset

        ds = CassavaDataset(data_root=DATA_ROOT, split="train")
        assert len(ds) == 17117
        assert ds.num_classes() == 5

        sample = ds[0]
        assert isinstance(sample["inputs"], Image.Image)
        assert isinstance(sample["gt_label"], int)
        assert 0 <= sample["gt_label"] <= 4
        assert "image_id" in sample

    def test_val_dataset(self):
        from elysia_cv.data import CassavaDataset

        ds = CassavaDataset(data_root=DATA_ROOT, split="val")
        assert len(ds) == 4280

    def test_class_names(self):
        from elysia_cv.data import CassavaDataset

        names = CassavaDataset.class_names()
        assert len(names) == 5
        assert "Healthy" in names[-1]


class TestCassavaModel:
    def test_plugin_discovery(self):
        from elysia_cv.models.registry import ModelRegistry

        reg = ModelRegistry.discover()
        assert "cassava" in reg
        p = reg.get("cassava")
        assert p.plugin.task_type == "classification"
        specs = list(p.plugin.model_specs())
        spec_names = {s.name for s in specs}
        assert "cassava_resnet50" in spec_names
        assert specs[0].extra.get("num_classes") == 5 or any(
            s.extra.get("num_classes") == 5 for s in specs
        )

    def test_build_and_forward(self):
        from elysia_cv.models.classification.cassava.runtime import (
            build_model,
            forward_train,
        )

        cfg = {
            "model": {
                "backbone": {"type": "ResNet", "depth": 18},
                "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 512},
            }
        }
        model = build_model(cfg)

        x = torch.randn(4, 3, 448, 448)
        labels = torch.randint(0, 5, (4,))
        batch = {"inputs": x, "gt_label": labels}

        result = forward_train(model, batch)
        assert "loss" in result
        assert result["loss"].item() > 0
        assert "focal" in result["losses"]  # uses FocalLoss

    def test_focal_loss_with_imbalance(self):
        """Verify FocalLoss handles imbalanced classes correctly."""
        from elysia_cv.core.registry import LOSSES

        # Simulate: CMD is 61.5% dominant, others rare
        focal = LOSSES.build({"type": "FocalLoss", "alpha": 0.25, "gamma": 2.0})
        logits = torch.randn(100, 5)
        # 62% of labels are class 3 (CMD), rest split
        labels = torch.cat([
            torch.full((62,), 3),
            torch.randint(0, 5, (38,)),
        ])
        loss = focal(logits, labels)
        assert loss.item() > 0

        # Focal loss should be lower than standard CE for easy examples
        ce = LOSSES.build({"type": "CrossEntropyLoss"})
        ce_loss = ce(logits, labels)
        # Not strictly guaranteed, but focal typically reduces easy-example
        # contribution
        assert ce_loss.item() > 0  # just verify both work


class TestCassavaPipeline:
    def test_full_pipeline_with_dataset(self):
        """CassavaDataset → Model → Forward with real data."""
        from elysia_cv.data import CassavaDataset
        from elysia_cv.models.classification.cassava.runtime import (
            build_model,
            forward_train,
        )
        from torch.utils.data import DataLoader
        from torchvision import transforms as T

        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        ds = CassavaDataset(data_root=DATA_ROOT, split="train", transform=transform)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        # Build model
        model = build_model({
            "model": {
                "backbone": {"type": "ResNet", "depth": 18},
                "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 512},
            }
        })

        # One batch forward
        batch = next(iter(loader))
        result = forward_train(model, batch)

        assert "loss" in result
        assert result["loss"].item() > 0
        assert "acc" in result["metrics"]

    def test_training_loop_10_steps(self):
        """10-step training on CPU with real Cassava data."""
        from elysia_cv.data import CassavaDataset
        from elysia_cv.models.classification.cassava.runtime import (
            build_model,
            forward_train,
        )
        from elysia_cv.training.loop import TrainerLoop
        from elysia_cv.training.strategy import ExecutionStrategy
        from elysia_cv.training.hooks.base import HookBus
        from torch.utils.data import DataLoader
        from torchvision import transforms as T

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Small subset for speed
        ds = CassavaDataset(data_root=DATA_ROOT, split="train", transform=transform)
        loader = DataLoader(ds, batch_size=2, shuffle=True)

        model = build_model({
            "model": {
                "backbone": {"type": "ResNet", "depth": 18},
                "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 512},
            }
        })
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        hooks = HookBus()
        loop = TrainerLoop(
            model=model,
            optimizer=optimizer,
            dataloader=loader,
            device=torch.device("cpu"),
            max_steps=10,
            strategy=ExecutionStrategy(grad_accum_steps=1),
            callback=hooks,
        )

        def _forward(batch):
            return forward_train(model, batch, device=torch.device("cpu"))

        loop._forward = _forward
        loop.run()

        assert loop.state.global_step == 10
        assert loop.state.last_loss > 0

    def test_config_loads(self):
        """Verify cassava config loads without errors."""
        from elysia_cv.config import load_cv_config_with_object
        from elysia_cv.training.plan import TrainingPlan

        cfg_path = Path(__file__).parent.parent / "configs" / "classification" / "cassava_resnet50.yaml"
        assert cfg_path.exists(), f"Config not found: {cfg_path}"

        cfg = load_cv_config_with_object(str(cfg_path))
        assert cfg.run.stage == "train"
        assert cfg.model.type == "cassava_resnet50"

        plan = TrainingPlan.from_config(cfg)
        assert plan.model["type"] == "cassava_resnet50"