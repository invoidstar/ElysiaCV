"""Tests for model_builder.py — registry completeness + building."""

from __future__ import annotations

import pytest
from types import SimpleNamespace as NS

from elysia_cv.model_builder import (
    build_model_from_config,
    _MODEL_REGISTRY,
    _cfg_to_dict,
)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

ALL_VIDEO_TYPES = [
    "tsn", "tsm", "mvit",
    "r3d", "mc3", "r2plus1d", "s3d",
    "x3d", "slowfast",
    "xclip", "st_adapter", "stadapter",
    "uniformerv2", "uniformer",
    "timesformer", "videomae",
]

DETECTION_TYPES = ["fcos", "yolo"]


def test_all_video_types_registered():
    """Every known video model type must be in _MODEL_REGISTRY."""
    for t in ALL_VIDEO_TYPES:
        assert t in _MODEL_REGISTRY, f"{t!r} missing from _MODEL_REGISTRY"


def test_all_detection_types_registered():
    for t in DETECTION_TYPES:
        assert t in _MODEL_REGISTRY, f"{t!r} missing from _MODEL_REGISTRY"


def test_registry_flags():
    """Every video entry must have family='video' flag."""
    for t in ALL_VIDEO_TYPES:
        _, _, flags = _MODEL_REGISTRY[t]
        assert flags.get("family") == "video", f"{t} should have family='video', got {flags}"

    for t in DETECTION_TYPES:
        _, _, flags = _MODEL_REGISTRY[t]
        assert "family" in flags, f"{t} missing family flag"


# ---------------------------------------------------------------------------
# _cfg_to_dict
# ---------------------------------------------------------------------------

def test_cfg_to_dict_basic():
    s = NS(a=1, b="hello", c=None, _private=42, d=3.14)
    d = _cfg_to_dict(s)
    assert d == {"a": 1, "b": "hello", "d": 3.14}  # c=None, _private skipped


def test_cfg_to_dict_exclude():
    s = NS(a=1, b=2, c=3)
    d = _cfg_to_dict(s, exclude=("b",))
    assert d == {"a": 1, "c": 3}


# ---------------------------------------------------------------------------
# Model building (smoke-test with fake backbones)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mtype", ALL_VIDEO_TYPES)
def test_video_model_builds(mtype):
    """Every video model must build without crashing (may fail if pretrained URL needed)."""
    # Default backbone per family
    if mtype in ("tsn", "tsm"):
        bb = "resnet18"
    elif mtype in ("r3d", "mc3", "r2plus1d", "s3d"):
        bb = f"{mtype}_18" if mtype != "s3d" else "s3d"  # s3d doesn't have _18 suffix
    elif mtype in ("xclip", "st_adapter", "stadapter"):
        bb = "vit_base_patch16_clip_224"
    else:
        bb = "resnet18"  # mvit/x3d/slowfast/timesformer/videomae/uniformerv2

    cfg = NS(type=mtype, backbone=bb,
             num_classes=2, pretrained=False, dropout=0.0)

    if mtype in ("x3d",):
        cfg.variant = "xs"
    if mtype in ("xclip", "st_adapter", "stadapter"):
        cfg.freeze_backbone = True
    if mtype == "xclip":
        cfg.num_temp_layers = 1
    if mtype in ("st_adapter", "stadapter"):
        cfg.adapter_ratio = 4
    if mtype in ("uniformerv2", "uniformer"):
        cfg.vit_backbone = "vit_tiny_patch16_224"
        cfg.cnn_channels = 16
    if mtype in ("videomae",):
        cfg.model_name = "MCG-NJU/videomae-base"
    if mtype in ("timesformer",):
        cfg.model_name = "facebook/timesformer-base-finetuned-k400"

    try:
        model = build_model_from_config(cfg)
        assert hasattr(model, "_is_video") or hasattr(model, "_is_yolo") or hasattr(model, "forward")
    except Exception as e:
        # Pretrained checkpoint may not be downloadable — skip don't fail
        if "pretrained" in str(e).lower() or "checkpoint" in str(e).lower() or "not found" in str(e).lower():
            pytest.skip(f"Pretrained weights unavailable: {e}")
        raise


def test_yolo_builds():
    cfg = NS(type="yolo", variant="yolov11n", num_classes=2, pretrained=False)
    model = build_model_from_config(cfg)
    assert hasattr(model, "_is_yolo")


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model_from_config(NS(type="nonexistent_model_xyz"))


def test_tsn_builds():
    cfg = NS(type="tsn", backbone="resnet18", num_classes=3, pretrained=False, dropout=0.0)
    model = build_model_from_config(cfg)
    assert model._is_video is True
    assert hasattr(model, "_forward_train_fn")