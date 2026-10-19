"""P5 — Detection & Segmentation end-to-end tests."""

from __future__ import annotations

import torch

import elysia_cv  # noqa: F401 — triggers registry population


class TestDetectionPipeline:
    """End-to-end pipeline for Faster R-CNN detection."""

    def test_fpn_build_and_forward(self):
        """Build FPN and run forward with multi-scale features."""
        from elysia_cv.core.registry import NECKS

        fpn = NECKS.build({
            "type": "FPN",
            "in_channels": [256, 512, 1024, 2048],
            "out_channels": 256,
            "num_outs": 5,
        })

        feats = [
            torch.randn(2, 256, 200, 200),
            torch.randn(2, 512, 100, 100),
            torch.randn(2, 1024, 50, 50),
            torch.randn(2, 2048, 25, 25),
        ]
        outs = fpn(feats)
        assert len(outs) == 5
        for i, o in enumerate(outs):
            assert o.size(1) == 256, f"level {i}: wrong channels {o.size(1)}"

    def test_det_head_forward(self):
        """Build DetHead and run forward."""
        from elysia_cv.core.registry import HEADS

        head = HEADS.build({"type": "DetHead", "num_classes": 80})
        feats = [torch.randn(2, 256, 50, 50), torch.randn(2, 256, 25, 25)]
        outs = head(feats)
        assert "cls_score" in outs
        assert "bbox_pred" in outs
        assert len(outs["cls_score"]) == 2
        assert outs["cls_score"][0].size(1) == 80  # num_classes
        assert outs["bbox_pred"][0].size(1) == 4  # bbox dims

    def test_det_model_build_and_train(self):
        """Build FCOS + run forward_train."""
        from elysia_cv.models.detection.fcos.runtime import (
            build_model,
            forward_train,
        )
        from elysia_cv.core.registry import BACKBONES

        # Need backbone with multi-scale output
        cfg = {
            "model": {
                "backbone": {"type": "ResNet", "depth": 18, "out_indices": [0, 1, 2, 3],
                             "pool_last": False},
                "neck": {"type": "FPN", "in_channels": [64, 64, 128, 256],
                         "out_channels": 256, "num_outs": 5},
                "head": {"type": "DetHead", "num_classes": 20, "in_channels": 256},
            }
        }
        model = build_model(cfg)

        x = torch.randn(2, 3, 224, 224)
        # Provide mock bboxes (xyxy format, variable per image)
        batch = {
            "inputs": x,
            "gt_bboxes": [torch.rand(3, 4) * 200 for _ in range(2)],
            "gt_labels": [torch.randint(0, 20, (3,)) for _ in range(2)],
        }

        result = forward_train(model, batch)
        assert "loss" in result
        assert result["loss"].item() > 0
        assert "cls_loss" in result["losses"]
        assert "reg_loss" in result["losses"]

    def test_det_losses(self):
        """Verify detection loss functions."""
        from elysia_cv.core.registry import LOSSES

        smooth_l1 = LOSSES.build({"type": "SmoothL1Loss"})
        pred = torch.randn(8, 4)
        tgt = torch.randn(8, 4)
        loss = smooth_l1(pred, tgt)
        assert loss.item() > 0

        iou = LOSSES.build({"type": "IoULoss", "mode": "iou"})
        pred_boxes = torch.tensor([[0., 0., 10., 10.]] * 8) + torch.randn(8, 4) * 0.5
        tgt_boxes = torch.tensor([[0., 0., 10., 10.]] * 8)
        loss2 = iou(pred_boxes, tgt_boxes)
        assert loss2.item() > 0

        giou = LOSSES.build({"type": "GIoULoss"})
        loss3 = giou(pred_boxes, tgt_boxes)
        assert loss3.item() > 0

    def test_det_plugin_discovery(self):
        """fcos plugin auto-discovered with correct task_type."""
        from elysia_cv.models.registry import ModelRegistry

        reg = ModelRegistry.discover()
        assert "fcos" in reg
        p = reg.get("fcos")
        assert p.plugin.task_type == "detection"
        specs = list(p.plugin.model_specs())
        assert len(specs) == 2
        assert specs[0].name == "fcos_r50"


class TestSegmentationPipeline:
    """End-to-end pipeline for FCN segmentation."""

    def test_seg_head_forward(self):
        """Build SegHead and run forward."""
        from elysia_cv.core.registry import HEADS

        head = HEADS.build({"type": "SegHead", "num_classes": 150, "in_channels": 2048})
        x = torch.randn(2, 2048, 16, 16)
        out = head(x)
        assert out.shape == (2, 150, 16, 16)

    def test_seg_head_with_multi_feats(self):
        """SegHead selects the right feature by in_index."""
        from elysia_cv.core.registry import HEADS

        head = HEADS.build({"type": "SegHead", "num_classes": 10, "in_channels": 512, "in_index": 1})
        feats = [torch.randn(2, 256, 32, 32), torch.randn(2, 512, 16, 16)]
        out = head(feats)
        assert out.shape == (2, 10, 16, 16)

    def test_seg_model_build_and_train(self):
        """Build FCN + run forward_train."""
        from elysia_cv.models.segmentation.fcn.runtime import build_model, forward_train

        cfg = {
            "model": {
                "backbone": {"type": "ResNet", "depth": 18, "pool_last": False},
                "neck": {"type": "IdentityNeck"},
                "head": {"type": "SegHead", "num_classes": 20, "in_channels": 512},
            }
        }
        model = build_model(cfg)

        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 20, (2, 224, 224))
        batch = {"inputs": x, "gt_semantic_seg": target}

        result = forward_train(model, batch)
        assert "loss" in result
        assert result["loss"].item() > 0
        assert "seg_loss" in result["losses"]

    def test_seg_losses(self):
        """Verify segmentation loss functions."""
        from elysia_cv.core.registry import LOSSES

        dice = LOSSES.build({"type": "DiceLoss"})
        pred = torch.randn(2, 5, 32, 32)
        tgt = torch.randint(0, 5, (2, 32, 32))
        loss = dice(pred, tgt)
        assert loss.item() > 0

        combined = LOSSES.build({"type": "CombinedSegLoss"})
        loss2 = combined(pred, tgt)
        assert loss2.item() > 0

    def test_seg_metrics(self):
        """Verify segmentation metrics."""
        from elysia_cv.core.registry import METRICS

        miou = METRICS.build({"type": "MeanIoU", "num_classes": 5})
        pred = torch.randn(2, 5, 32, 32)
        tgt = torch.randint(0, 5, (2, 32, 32))
        miou.update(pred, tgt)
        result = miou.compute()
        assert "mIoU" in result
        assert 0.0 <= result["mIoU"] <= 1.0

        macc = METRICS.build({"type": "MeanAccuracy", "num_classes": 5})
        macc.update(pred, tgt)
        r2 = macc.compute()
        assert "mAcc" in r2

    def test_seg_plugin_discovery(self):
        """fcn plugin auto-discovered with correct task_type."""
        from elysia_cv.models.registry import ModelRegistry

        reg = ModelRegistry.discover()
        assert "fcn" in reg
        p = reg.get("fcn")
        assert p.plugin.task_type == "segmentation"
        specs = list(p.plugin.model_specs())
        assert len(specs) == 2
        assert specs[0].name == "fcn_r50"

    def test_det_seg_training_loop(self):
        """10-step training for both detection and segmentation on CPU."""
        from elysia_cv.training.loop import TrainerLoop
        from elysia_cv.training.strategy import ExecutionStrategy
        from elysia_cv.training.hooks.base import HookBus

        for (build_fn, train_fn, batch_fn, label_key) in [
            (
                self._build_det,
                self._det_forward,
                lambda: {
                    "inputs": torch.randn(3, 224, 224),
                    "gt_bboxes": torch.rand(2, 4) * 200,
                    "gt_labels": torch.randint(0, 5, (2,)),
                },
                None,
            ),
            (
                self._build_seg,
                self._seg_forward,
                lambda: {
                    "inputs": torch.randn(3, 224, 224),
                    "gt_semantic_seg": torch.randint(0, 5, (224, 224)),
                },
                None,
            ),
        ]:
            model = build_fn()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            class FakeDS:
                def __init__(self): pass
                def __len__(self): return 32
                def __getitem__(self, idx): return batch_fn()

            from torch.utils.data import DataLoader
            loader = DataLoader(FakeDS(), batch_size=2, shuffle=True)

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
            loop._forward = lambda batch: train_fn(model, batch, device=torch.device("cpu"))
            loop.run()

            assert loop.state.global_step == 10

    @staticmethod
    def _build_det():
        from elysia_cv.models.detection.fcos.runtime import build_model
        return build_model({"model": {
            "backbone": {"type": "ResNet", "depth": 18, "out_indices": [0, 1, 2, 3]},
            "neck": {"type": "FPN", "in_channels": [64, 64, 128, 256],
                     "out_channels": 256, "num_outs": 5},
            "head": {"type": "DetHead", "num_classes": 5, "in_channels": 256},
        }})

    @staticmethod
    def _det_forward(model, batch, *, device=None):
        from elysia_cv.models.detection.fcos.runtime import forward_train
        return forward_train(model, batch, device=device)

    @staticmethod
    def _build_seg():
        from elysia_cv.models.segmentation.fcn.runtime import build_model
        return build_model({"model": {
            "backbone": {"type": "ResNet", "depth": 18, "pool_last": False},
            "neck": {"type": "IdentityNeck"},
            "head": {"type": "SegHead", "num_classes": 5, "in_channels": 512},
        }})

    @staticmethod
    def _seg_forward(model, batch, *, device=None):
        from elysia_cv.models.segmentation.fcn.runtime import forward_train
        return forward_train(model, batch, device=device)


class TestNeckComponent:
    def test_identity_neck_passthrough(self):
        from elysia_cv.core.registry import NECKS
        neck = NECKS.build({"type": "IdentityNeck"})
        x = torch.randn(2, 512, 16, 16)
        out = neck(x)
        assert torch.equal(out, x)

    def test_identity_neck_list_passthrough(self):
        from elysia_cv.core.registry import NECKS
        neck = NECKS.build({"type": "IdentityNeck"})
        feats = [torch.randn(2, 256, 8, 8), torch.randn(2, 512, 4, 4)]
        out = neck(feats)
        assert len(out) == 2
        for a, b in zip(out, feats):
            assert torch.equal(a, b)