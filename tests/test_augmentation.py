"""P6 — Data augmentation tests: MixUp, CutMix, Mosaic, geometric transforms."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import elysia_cv  # noqa: F401


# ── Helpers ─────────────────────────────────────────────────


def _make_img_batch(b: int = 4, h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.randn(b, 3, h, w)


def _make_labels(b: int = 4, num_classes: int = 5) -> torch.Tensor:
    return torch.randint(0, num_classes, (b,))


# ── MixUp ───────────────────────────────────────────────────


class TestMixUp:
    """Batch-level MixUp augmentation."""

    def test_basic_mixing(self):
        from elysia_cv.data.transforms.augmentation import MixUp
        mixup = MixUp(alpha=0.4, prob=1.0, num_classes=5)
        batch = {"inputs": _make_img_batch(), "gt_label": _make_labels()}
        orig = batch["inputs"].clone()

        result = mixup(batch)
        assert "gt_label_soft" in result
        assert "gt_label_orig" in result
        assert "mix_lam" in result
        # Images must differ
        assert not torch.allclose(result["inputs"], orig)
        # Soft labels sum to 1 per sample
        assert torch.allclose(result["gt_label_soft"].sum(dim=1), torch.ones(4))

    def test_disabled_by_prob(self):
        from elysia_cv.data.transforms.augmentation import MixUp
        mixup = MixUp(alpha=0.4, prob=0.0, num_classes=5)
        batch = {"inputs": _make_img_batch(), "gt_label": _make_labels()}
        result = mixup(batch)
        assert "gt_label_soft" not in result

    def test_disabled_by_alpha_zero(self):
        from elysia_cv.data.transforms.augmentation import MixUp
        mixup = MixUp(alpha=0.0, prob=1.0, num_classes=5)
        batch = {"inputs": _make_img_batch(), "gt_label": _make_labels()}
        result = mixup(batch)
        assert "gt_label_soft" not in result

    def test_soft_label_shape(self):
        from elysia_cv.data.transforms.augmentation import MixUp
        mixup = MixUp(alpha=0.2, prob=1.0, num_classes=10)
        batch = {"inputs": _make_img_batch(8, 32, 32), "gt_label": torch.randint(0, 10, (8,))}
        result = mixup(batch)
        assert result["gt_label_soft"].shape == (8, 10)


# ── CutMix ──────────────────────────────────────────────────


class TestCutMix:
    """Batch-level CutMix augmentation."""

    def test_basic_cutmix(self):
        from elysia_cv.data.transforms.augmentation import CutMix
        cutmix = CutMix(alpha=1.0, prob=1.0, num_classes=5)
        batch = {"inputs": _make_img_batch(4, 128, 128), "gt_label": _make_labels()}
        orig = batch["inputs"].clone()

        result = cutmix(batch)
        assert "gt_label_soft" in result
        assert "gt_label_orig" in result
        # With high alpha and large image, cut region is significant
        assert not torch.allclose(result["inputs"], orig, atol=1e-5)

    def test_cutmix_disabled_by_alpha(self):
        from elysia_cv.data.transforms.augmentation import CutMix
        cutmix = CutMix(alpha=0.0, prob=1.0, num_classes=5)
        result = cutmix({"inputs": _make_img_batch(4, 64, 64), "gt_label": _make_labels()})
        # alpha=0 disables the transform → no soft labels
        assert "gt_label_soft" not in result

    def test_cutmix_disabled_by_prob(self):
        from elysia_cv.data.transforms.augmentation import CutMix
        cutmix = CutMix(alpha=1.0, prob=0.0, num_classes=5)
        result = cutmix({"inputs": _make_img_batch(4, 64, 64), "gt_label": _make_labels()})
        assert "gt_label_soft" not in result

    def test_lam_in_range(self):
        from elysia_cv.data.transforms.augmentation import CutMix
        cutmix = CutMix(alpha=1.0, prob=1.0, num_classes=5)
        result = cutmix({"inputs": _make_img_batch(), "gt_label": _make_labels()})
        assert 0.0 <= result["mix_lam"] <= 1.0


# ── Mosaic ──────────────────────────────────────────────────


class TestMosaic:
    """Batch-level Mosaic detection augmentation."""

    def test_four_to_one(self):
        from elysia_cv.data.transforms.augmentation import Mosaic
        mosaic = Mosaic(output_size=(320, 320), prob=1.0)
        batch = {
            "inputs": _make_img_batch(4, 200, 300),
            "gt_bboxes": [torch.zeros(1, 4) for _ in range(4)],
            "gt_labels": [torch.zeros(1, dtype=torch.long) for _ in range(4)],
        }
        result = mosaic(batch)
        assert result["inputs"].shape[0] == 1
        assert result["inputs"].shape[1:] == (3, 320, 320)

    def test_eight_to_two(self):
        from elysia_cv.data.transforms.augmentation import Mosaic
        mosaic = Mosaic(output_size=(640, 480), prob=1.0)
        batch = {
            "inputs": _make_img_batch(8, 100, 100),
            "gt_bboxes": [torch.ones(1, 4) for _ in range(8)],
            "gt_labels": [torch.zeros(1, dtype=torch.long) for _ in range(8)],
        }
        result = mosaic(batch)
        assert result["inputs"].shape[0] == 2

    def test_batch_less_than_four_noop(self):
        from elysia_cv.data.transforms.augmentation import Mosaic
        mosaic = Mosaic(prob=1.0)
        batch = {"inputs": _make_img_batch(3, 50, 50)}
        result = mosaic(batch)
        assert result["inputs"].shape[0] == 3

    def test_disabled_by_prob(self):
        from elysia_cv.data.transforms.augmentation import Mosaic
        mosaic = Mosaic(prob=0.0)
        batch = {"inputs": _make_img_batch(4, 50, 50)}
        result = mosaic(batch)
        assert result["inputs"].shape[0] == 4

    def test_bbox_count_preserved(self):
        from elysia_cv.data.transforms.augmentation import Mosaic
        mosaic = Mosaic(output_size=(320, 320), prob=1.0)
        bboxes = [
            torch.tensor([[10., 20., 80., 90.], [100., 50., 180., 140.]]),  # 2
            torch.tensor([[30., 40., 150., 120.]]),                           # 1
            torch.tensor([[5., 5., 50., 60.]]),                               # 1
            torch.tensor([[40., 30., 160., 170.]]),                           # 1
        ]
        batch = {
            "inputs": _make_img_batch(4, 200, 300),
            "gt_bboxes": bboxes,
            "gt_labels": [torch.randint(0, 3, (b.shape[0],)) for b in bboxes],
        }
        result = mosaic(batch)
        total = sum(b.shape[0] for b in result["gt_bboxes"])
        assert total == 5  # 2+1+1+1


# ── Geometric transforms ────────────────────────────────────


class TestGeometricTransforms:
    """RandomRotation + RandomAffine with and without bboxes."""

    def test_rotation_changes_shape(self):
        from elysia_cv.data.transforms.augmentation import RandomRotation
        rot = RandomRotation(degrees=30, prob=1.0, expand=True)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = rot({"img": img})
        assert result["img"].shape[:2] != (64, 64)  # expanded

    def test_rotation_preserves_bbox_count(self):
        from elysia_cv.data.transforms.augmentation import RandomRotation
        rot = RandomRotation(degrees=15, prob=1.0, expand=True)
        result = rot({
            "img": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            "gt_bboxes": np.array([[10, 20, 60, 80], [70, 50, 110, 100]], dtype=np.float32),
        })
        assert result["gt_bboxes"].shape == (2, 4)

    def test_rotation_empty_bboxes(self):
        from elysia_cv.data.transforms.augmentation import RandomRotation
        rot = RandomRotation(degrees=15, prob=1.0)
        result = rot({
            "img": np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8),
            "gt_bboxes": np.zeros((0, 4)),
        })
        assert result["gt_bboxes"].shape == (0, 4)

    def test_rotation_no_bboxes(self):
        from elysia_cv.data.transforms.augmentation import RandomRotation
        rot = RandomRotation(degrees=15, prob=1.0)
        result = rot({"img": np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)})
        assert "gt_bboxes" not in result

    def test_affine_changes_shape(self):
        from elysia_cv.data.transforms.augmentation import RandomAffine
        aff = RandomAffine(translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5.0, prob=1.0)
        result = aff({
            "img": np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8),
            "gt_bboxes": np.array([[15, 20, 50, 60]], dtype=np.float32),
        })
        assert result["gt_bboxes"].shape == (1, 4)

    def test_affine_empty_bboxes(self):
        from elysia_cv.data.transforms.augmentation import RandomAffine
        aff = RandomAffine(prob=1.0)
        result = aff({
            "img": np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8),
            "gt_bboxes": np.zeros((0, 4)),
        })
        assert result["gt_bboxes"].shape == (0, 4)

    def test_affine_deterministic(self):
        """Affine with zero jitter should not crash."""
        from elysia_cv.data.transforms.augmentation import RandomAffine
        aff = RandomAffine(translate=(0.0, 0.0), scale=(1.0, 1.0), shear=0.0, prob=1.0)
        result = aff({"img": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)})
        assert result["img"].shape == (64, 64, 3)


# ── Pipeline ────────────────────────────────────────────────


class TestAugPipeline:
    """Compose + batch pipeline integration."""

    def test_build_pipeline_with_rotation(self):
        from elysia_cv.data.transforms import build_pipeline
        pipe = build_pipeline([
            {"type": "RandomRotation", "degrees": 10, "prob": 0.5},
        ])
        result = pipe({"img": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)})
        assert "img" in result

    def test_batch_pipeline_mixup_cutmix(self):
        from elysia_cv.data.transforms import build_batch_pipeline
        batch_tfs = build_batch_pipeline([
            {"type": "MixUp", "alpha": 0.2, "prob": 1.0, "num_classes": 3},
            {"type": "CutMix", "alpha": 1.0, "prob": 1.0, "num_classes": 3},
        ])
        assert len(batch_tfs) == 2
        batch = {"inputs": _make_img_batch(), "gt_label": _make_labels(num_classes=3)}
        for tf in batch_tfs:
            batch = tf(batch)
        assert "gt_label_soft" in batch

    def test_compose_with_no_batch_transforms(self):
        from elysia_cv.data.transforms import Compose
        pipe = Compose([])
        result = pipe({"img": np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)})
        assert "img" in result
        # apply_batch with no transforms is a no-op
        b = pipe.apply_batch({"inputs": _make_img_batch(2, 32, 32)})
        assert b["inputs"].shape == (2, 3, 32, 32)