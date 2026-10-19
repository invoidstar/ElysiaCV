"""Cassava experiment comparison — 6 architectures on fine-grained classification."""

from __future__ import annotations

import torch
from pathlib import Path

import elysia_cv  # noqa: F401

DATA_ROOT = "/data/hdd3/chuqi/data/cassava"

# ── Experiment definitions ─────────────────────────────────────────────
EXPERIMENTS = {
    "mlp": {
        "type": "cassava_mlp",
        "backbone": {"type": "MLP", "input_size": [224, 224],
                      "hidden_dims": [1024, 512], "dropout": 0.3},
        "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 512, "dropout": 0.2},
        "batch": 64,
        "opt": "AdamW",
        "lr": 0.001,
    },
    "simple_cnn": {
        "type": "cassava_simple_cnn",
        "backbone": {"type": "SimpleCNN", "base_channels": 32,
                      "num_classes": 512, "dropout": 0.3},
        "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 512, "dropout": 0.2},
        "batch": 32,
        "opt": "SGD",
        "lr": 0.01,
    },
    "resnet18": {
        "type": "cassava_resnet18",
        "backbone": {"type": "ResNet", "depth": 18, "pool_last": True},
        "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 512, "dropout": 0.2},
        "batch": 32,
        "opt": "SGD",
        "lr": 0.01,
    },
    "resnet34": {
        "type": "cassava_resnet34",
        "backbone": {"type": "ResNet", "depth": 34, "pool_last": True},
        "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 512, "dropout": 0.2},
        "batch": 32,
        "opt": "SGD",
        "lr": 0.01,
    },
    "vit_b16": {
        "type": "cassava_vit_b16",
        "backbone": {"type": "ViT", "variant": "vit_b_16", "image_size": 224,
                      "pretrained": False, "dropout": 0.1},
        "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 768, "dropout": 0.1},
        "batch": 16,
        "opt": "AdamW",
        "lr": 0.0003,
    },
    "vit_l16": {
        "type": "cassava_vit_l16",
        "backbone": {"type": "ViT", "variant": "vit_l_16", "image_size": 224,
                      "pretrained": False, "dropout": 0.1},
        "head": {"type": "LinearClsHead", "num_classes": 5, "in_channels": 1024, "dropout": 0.1},
        "batch": 8,
        "opt": "AdamW",
        "lr": 0.0001,
    },
}


class TestBackbones:
    """Verify all 4 backbone types register and forward."""

    def test_mlp_forward(self):
        from elysia_cv.core.registry import BACKBONES
        m = BACKBONES.build({"type": "MLP", "input_size": [224, 224],
                              "hidden_dims": [512, 256]})
        out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 256)

    def test_simple_cnn_forward(self):
        from elysia_cv.core.registry import BACKBONES
        m = BACKBONES.build({"type": "SimpleCNN", "base_channels": 32, "num_classes": 256})
        out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 256)

    def test_vit_b16_forward(self):
        from elysia_cv.core.registry import BACKBONES
        m = BACKBONES.build({"type": "ViT", "variant": "vit_b_16", "image_size": 224})
        out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 768)

    def test_vit_l16_forward(self):
        from elysia_cv.core.registry import BACKBONES
        m = BACKBONES.build({"type": "ViT", "variant": "vit_l_16", "image_size": 224})
        out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 1024)

    def test_resnet_unchanged(self):
        from elysia_cv.core.registry import BACKBONES
        m = BACKBONES.build({"type": "ResNet", "depth": 18})
        out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 512)


class TestModelRegistry:
    """Verify all 6 cassava specs are registered."""

    def test_all_specs_discovered(self):
        from elysia_cv.models.registry import ModelRegistry
        reg = ModelRegistry.discover()
        assert "cassava" in reg

        p = reg.get("cassava")
        spec_names = {s.name for s in p.plugin.model_specs()}
        expected = {
            "cassava_mlp", "cassava_simple_cnn",
            "cassava_resnet18", "cassava_resnet34", "cassava_resnet50",
            "cassava_vit_b16", "cassava_vit_l16",
            "cassava_herbs_swin_base", "cassava_herbs_swin_tiny",
        }
        assert spec_names == expected


def _build_experiment(name: str):
    from elysia_cv.models.classification.cassava.runtime import build_model
    spec = EXPERIMENTS[name]
    return build_model({"model": {**spec}})


class TestBuildAndForward:
    """Build each model and run one forward pass."""

    def _test_single(self, name: str):
        model = _build_experiment(name)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 5)

    def test_build_mlp(self):
        self._test_single("mlp")

    def test_build_simple_cnn(self):
        self._test_single("simple_cnn")

    def test_build_resnet18(self):
        self._test_single("resnet18")

    def test_build_resnet34(self):
        self._test_single("resnet34")

    def test_build_vit_b16(self):
        self._test_single("vit_b16")

    def test_build_vit_l16(self):
        self._test_single("vit_l16")

    def _test_train_single(self, name: str):
        from elysia_cv.models.classification.cassava.runtime import forward_train
        model = _build_experiment(name)
        batch = {"inputs": torch.randn(2, 3, 224, 224),
                 "gt_label": torch.randint(0, 5, (2,))}
        result = forward_train(model, batch)
        assert "loss" in result
        assert result["loss"].item() > 0
        assert "acc" in result["metrics"]

    def test_train_mlp(self):
        self._test_train_single("mlp")

    def test_train_simple_cnn(self):
        self._test_train_single("simple_cnn")

    def test_train_resnet18(self):
        self._test_train_single("resnet18")

    def test_train_resnet34(self):
        self._test_train_single("resnet34")

    def test_train_vit_b16(self):
        self._test_train_single("vit_b16")

    def test_train_vit_l16(self):
        self._test_train_single("vit_l16")


class TestConfigs:
    """Verify all 7 YAML configs load without error."""

    _CONFIG_DIR = Path(__file__).parent.parent / "configs" / "classification"

    def _check_config(self, name: str):
        from elysia_cv.config import load_cv_config_with_object
        from elysia_cv.training.plan import TrainingPlan
        cfg_path = self._CONFIG_DIR / f"cassava_{name}.yaml"
        assert cfg_path.exists(), f"Missing config: {cfg_path}"
        cfg = load_cv_config_with_object(str(cfg_path))
        assert cfg.run.stage == "train"
        assert cfg.model.type.startswith("cassava_")
        TrainingPlan.from_config(cfg)

    def test_config_mlp(self):
        self._check_config("mlp")

    def test_config_simple_cnn(self):
        self._check_config("simple_cnn")

    def test_config_resnet18(self):
        self._check_config("resnet18")

    def test_config_resnet34(self):
        self._check_config("resnet34")

    def test_config_resnet50(self):
        self._check_config("resnet50")

    def test_config_vit_b16(self):
        self._check_config("vit_b16")

    def test_config_vit_l16(self):
        self._check_config("vit_l16")


class TestComparison10Step:
    """10-step training loop — all 6 architectures."""

    def _run_10_steps(self, name: str):
        from elysia_cv.data import CassavaDataset
        from elysia_cv.models.classification.cassava.runtime import (
            build_model, forward_train,
        )
        from elysia_cv.training.loop import TrainerLoop
        from elysia_cv.training.strategy import ExecutionStrategy
        from elysia_cv.training.hooks.base import HookBus
        from torch.utils.data import DataLoader
        from torchvision import transforms as T

        spec = EXPERIMENTS[name]
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        ds = CassavaDataset(data_root=DATA_ROOT, split="train", transform=transform)
        loader = DataLoader(ds, batch_size=min(spec["batch"], 4), shuffle=True)
        model = build_model({"model": {**spec}})
        opt = (torch.optim.AdamW(model.parameters(), lr=spec["lr"])
               if spec["opt"] == "AdamW"
               else torch.optim.SGD(model.parameters(), lr=spec["lr"], momentum=0.9))

        hooks = HookBus()
        loop = TrainerLoop(
            model=model, optimizer=opt, dataloader=loader,
            device=torch.device("cpu"), max_steps=10,
            strategy=ExecutionStrategy(grad_accum_steps=1),
            callback=hooks,
        )
        def _fwd(batch):
            return forward_train(model, batch, device=torch.device("cpu"))
        loop._forward = _fwd
        loop.run()
        assert loop.state.global_step == 10
        assert loop.state.last_loss > 0

    def test_10step_mlp(self):
        self._run_10_steps("mlp")

    def test_10step_simple_cnn(self):
        self._run_10_steps("simple_cnn")

    def test_10step_resnet18(self):
        self._run_10_steps("resnet18")

    def test_10step_resnet34(self):
        self._run_10_steps("resnet34")

    def test_10step_vit_b16(self):
        self._run_10_steps("vit_b16")

    def test_10step_vit_l16(self):
        self._run_10_steps("vit_l16")