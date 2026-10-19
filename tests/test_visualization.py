"""P6 — Visualization tests: PR curves, GT/Pred comparison."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from PIL import Image

import elysia_cv  # noqa: F401


class TestPRCurves:
    """Precision-Recall curve extraction and plotting."""

    def test_get_pr_data_nonempty(self):
        from elysia_cv.metrics.iou import MeanAP

        metric = MeanAP(num_classes=3)
        for _ in range(4):
            metric.update(
                pred_boxes=torch.rand(3, 4) * 100,
                pred_scores=torch.rand(3),
                pred_labels=torch.randint(0, 3, (3,)),
                gt_boxes=torch.rand(2, 4) * 100,
                gt_labels=torch.randint(0, 3, (2,)),
            )
        _ = metric.compute()
        pr = metric.get_pr_data(iou_thr=0.5)

        assert len(pr) >= 1
        for cls_id, (prec, rec) in pr.items():
            assert len(prec) == 101
            assert len(rec) == 101
            # Precision values in [0, 1]
            assert all(0.0 <= p <= 1.0 for p in prec)

    def test_get_pr_data_empty(self):
        from elysia_cv.metrics.iou import MeanAP

        metric = MeanAP(num_classes=5)
        metric.compute()
        assert metric.get_pr_data() == {}

    def test_plot_pr_curves_saves_png(self):
        from elysia_cv.metrics.iou import MeanAP
        from elysia_cv.utils.visualization import plot_pr_curves

        metric = MeanAP(num_classes=2)
        for _ in range(3):
            metric.update(
                pred_boxes=torch.rand(2, 4) * 50,
                pred_scores=torch.rand(2),
                pred_labels=torch.randint(0, 2, (2,)),
                gt_boxes=torch.rand(1, 4) * 50,
                gt_labels=torch.randint(0, 2, (1,)),
            )
        _ = metric.compute()
        pr = metric.get_pr_data()

        with tempfile.TemporaryDirectory() as d:
            path = plot_pr_curves(pr, os.path.join(d, "pr.png"),
                                  class_names=["cat", "dog"])
            assert path and os.path.exists(path)
            assert os.path.getsize(path) > 1000

    def test_plot_empty_returns_none(self):
        from elysia_cv.utils.visualization import plot_pr_curves
        assert plot_pr_curves({}) is None

    def test_plot_default_path(self):
        from elysia_cv.metrics.iou import MeanAP
        from elysia_cv.utils.visualization import plot_pr_curves

        metric = MeanAP(num_classes=1)
        metric.update(
            pred_boxes=torch.rand(1, 4) * 50, pred_scores=torch.tensor([0.9]),
            pred_labels=torch.tensor([0]), gt_boxes=torch.rand(1, 4) * 50,
            gt_labels=torch.tensor([0]),
        )
        _ = metric.compute()
        pr = metric.get_pr_data()
        import shutil

        with tempfile.TemporaryDirectory() as root:
            orig_cwd = os.getcwd()
            try:
                os.chdir(root)
                path = plot_pr_curves(pr)
                assert path and os.path.exists(path)
            finally:
                os.chdir(orig_cwd)


class TestDetectionVisualization:
    """GT vs Prediction comparison drawing."""

    def test_gt_pred_comparison_creates_wide_image(self):
        from elysia_cv.utils.visualization import draw_gt_pred_comparison

        img = Image.fromarray(np.random.randint(0, 255, (160, 240, 3), dtype=np.uint8))
        gt_bboxes = np.array([[30, 40, 120, 130]], dtype=np.float32)
        gt_labels = np.array([0], dtype=np.int64)
        pred_bboxes = np.array([[35, 45, 125, 135]], dtype=np.float32)
        pred_scores = np.array([0.92], dtype=np.float32)
        pred_labels = np.array([0], dtype=np.int64)

        comp = draw_gt_pred_comparison(
            img, gt_bboxes, gt_labels,
            pred_bboxes, pred_scores, pred_labels,
            class_names=["person"],
        )
        # Composite should be wider than original (two panels + gap)
        assert comp.size[0] > img.size[0] * 1.5

    def test_gt_pred_saves_to_disk(self):
        from elysia_cv.utils.visualization import draw_gt_pred_comparison

        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "comp.png")
            comp = draw_gt_pred_comparison(
                img,
                np.array([[10, 10, 50, 60]], dtype=np.float32),
                np.array([1], dtype=np.int64),
                np.array([[12, 12, 52, 62]], dtype=np.float32),
                np.array([0.88], dtype=np.float32),
                np.array([1], dtype=np.int64),
                output_path=out,
            )
            assert os.path.exists(out)
            assert os.path.getsize(out) > 500

    def test_score_threshold_filters(self):
        from elysia_cv.utils.visualization import draw_gt_pred_comparison

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        comp = draw_gt_pred_comparison(
            img,
            np.array([[10, 10, 50, 50]], dtype=np.float32),
            np.array([0], dtype=np.int64),
            np.array([[10, 10, 50, 50]], dtype=np.float32),
            np.array([0.1], dtype=np.float32),  # below 0.3 threshold
            np.array([0], dtype=np.int64),
            score_thr=0.3,
        )
        # Should still create the composite even with no preds visible
        assert comp.size[0] > img.size[0]


class TestSegmentationVisualization:
    """Segmentation overlay drawing."""

    def test_basic_overlay(self):
        from elysia_cv.utils.visualization import draw_segmentation_overlay

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        seg_map = np.random.randint(0, 4, (64, 64))
        result = draw_segmentation_overlay(img, seg_map, alpha=0.5)
        assert isinstance(result, Image.Image)
        assert result.size == (64, 64)

    def test_numpy_input(self):
        from elysia_cv.utils.visualization import draw_segmentation_overlay

        img_np = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        seg_map = np.zeros((64, 64), dtype=int)
        seg_map[16:48, 16:48] = 1
        result = draw_segmentation_overlay(img_np, seg_map)
        assert result.size == (64, 64)


class TestBBoxDrawing:
    """Basic bbox drawing regression."""

    def test_draw_bboxes_rgb(self):
        from elysia_cv.utils.visualization import draw_bboxes

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = draw_bboxes(
            img,
            np.array([[10, 10, 80, 80]], dtype=np.float32),
            labels=np.array([0]),
            scores=np.array([0.95]),
            class_names=["dog"],
        )
        assert result.size == (100, 100)

    def test_draw_bboxes_numpy_input(self):
        from elysia_cv.utils.visualization import draw_bboxes

        img_np = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = draw_bboxes(img_np, np.array([[10, 10, 80, 80]], dtype=np.float32))
        assert isinstance(result, Image.Image)

    def test_score_threshold(self):
        from elysia_cv.utils.visualization import draw_bboxes

        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        # Should not crash with score == 0
        result = draw_bboxes(img, np.array([[10, 10, 50, 50]], dtype=np.float32),
                             scores=np.array([0.0]), score_thr=0.5)
        assert result.size == (100, 100)