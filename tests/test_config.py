"""Unit tests for the config system: loading, overrides, _base_ inheritance, validation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from elysia_cv.config.loader import (
    _parse_overrides,
    _parse_value,
    _deep_merge,
    _merge_list_by_index,
    expand_environment_values,
    load_cv_config,
    load_cv_config_with_object,
)
from elysia_cv.config.config import CVConfig
from elysia_cv.config.validation import validate_cross_section, validate_or_raise


# ---------------------------------------------------------------------------
# _parse_value tests
# ---------------------------------------------------------------------------


class TestParseValue:
    def test_null(self):
        assert _parse_value("null") is None
        assert _parse_value("None") is None

    def test_bool(self):
        assert _parse_value("true") is True
        assert _parse_value("false") is False
        assert _parse_value("True") is True
        assert _parse_value("FALSE") is False

    def test_int(self):
        assert _parse_value("42") == 42
        assert _parse_value("-5") == -5

    def test_float(self):
        assert _parse_value("3.14") == 3.14
        assert _parse_value("0.001") == 0.001

    def test_json(self):
        assert _parse_value('{"a": 1}') == {"a": 1}
        assert _parse_value("[1, 2, 3]") == [1, 2, 3]

    def test_comma_list(self):
        assert _parse_value("a,b,c") == ["a", "b", "c"]

    def test_string(self):
        assert _parse_value("hello world") == "hello world"


# ---------------------------------------------------------------------------
# _parse_overrides tests
# ---------------------------------------------------------------------------


class TestParseOverrides:
    def test_string_overrides(self):
        result = _parse_overrides(["model.type=resnet50", "schedule.base_lr=0.01"])
        assert result["model"]["type"] == "resnet50"
        assert result["schedule"]["base_lr"] == 0.01

    def test_mapping_overrides(self):
        result = _parse_overrides({"model.type": "vit", "optim.params.lr": 0.001})
        assert result["model"]["type"] == "vit"
        assert result["optim"]["params"]["lr"] == 0.001

    def test_nested_override(self):
        result = _parse_overrides(["engine.distributed.proc_per_node=4"])
        assert result["engine"]["distributed"]["proc_per_node"] == 4

    def test_unknown_key_fails(self):
        with pytest.raises(ValueError, match="unknown override key"):
            _parse_overrides(["model.typo=resnet50"])

    def test_unknown_top_level_fails(self):
        with pytest.raises(ValueError, match="unknown override key"):
            _parse_overrides(["bad_section.key=value"])


# ---------------------------------------------------------------------------
# _deep_merge tests
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        overrides = {"b": 99, "c": 3}
        assert _deep_merge(base, overrides) == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"engine": {"backend": "native", "distributed": {"proc_per_node": 1}}}
        overrides = {"engine": {"distributed": {"proc_per_node": 8}}}
        result = _deep_merge(base, overrides)
        assert result["engine"]["backend"] == "native"  # preserved
        assert result["engine"]["distributed"]["proc_per_node"] == 8  # overridden


# ---------------------------------------------------------------------------
# _merge_list_by_index tests
# ---------------------------------------------------------------------------


class TestMergeListByIndex:
    def test_index_merge(self):
        base = [{"name": "a"}, {"name": "b"}]
        overrides = {"0": {"name": "aa"}}
        result = _merge_list_by_index(base, overrides)
        assert result == [{"name": "aa"}, {"name": "b"}]

    def test_extend_list(self):
        base = [{"name": "a"}]
        overrides = {"2": {"name": "c"}}
        result = _merge_list_by_index(base, overrides)
        assert len(result) == 3
        assert result[2] == {"name": "c"}


# ---------------------------------------------------------------------------
# Environment expansion tests
# ---------------------------------------------------------------------------


class TestEnvExpansion:
    def test_no_env(self):
        assert expand_environment_values("hello") == "hello"

    def test_existing_env(self):
        os.environ["ELYSIA_CV_TEST_VAR"] = "world"
        try:
            assert expand_environment_values("${ELYSIA_CV_TEST_VAR}") == "world"
        finally:
            del os.environ["ELYSIA_CV_TEST_VAR"]

    def test_fallback(self):
        assert expand_environment_values("${MISSING:-default}") == "default"

    def test_nested(self):
        result = expand_environment_values({"a": "${MISSING:-x}", "b": [1, "c"]})
        assert result == {"a": "x", "b": [1, "c"]}


# ---------------------------------------------------------------------------
# _base_ inheritance tests
# ---------------------------------------------------------------------------


class TestBaseInheritance:
    def test_base_inheritance(self):
        config_dir = Path(__file__).parent
        # Use the existing default_runtime.yaml as base
        base_path = config_dir.parent / "configs" / "_base_" / "default_runtime.yaml"
        if not base_path.exists():
            pytest.skip("default_runtime.yaml not found")

        # Create temp child config that inherits from base
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            child_path = Path(tmp) / "child.yaml"
            child_path.write_text("_base_: " + json.dumps(str(base_path)))

            result = load_cv_config(child_path)
            assert isinstance(result, dict)
            assert "engine" in result  # from base


# ---------------------------------------------------------------------------
# Full load + CVConfig construction
# ---------------------------------------------------------------------------


class TestCVConfigLoad:
    CONFIG_DIR = Path(__file__).parent.parent / "configs"

    def test_load_classification_config(self):
        path = self.CONFIG_DIR / "classification" / "resnet50_imagenet.yaml"
        if not path.exists():
            pytest.skip(f"{path} not found")
        config = load_cv_config_with_object(path)
        assert config.run.stage == "train"
        assert config.model.type == "resnet50"
        assert config.data.train.dataset_name == "ImageNet"
        assert config.schedule.steps == 100000

    def test_load_detection_config(self):
        path = self.CONFIG_DIR / "detection" / "fcos_resnet50_coco.yaml"
        if not path.exists():
            pytest.skip(f"{path} not found")
        config = load_cv_config_with_object(path)
        assert config.model.type == "fcos"
        assert config.data.train.dataset_name == "COCO"
        assert config.eval.metrics[0]["type"] == "MeanAP"

    def test_load_segmentation_config(self):
        path = self.CONFIG_DIR / "segmentation" / "fcn_resnet50_ade20k.yaml"
        if not path.exists():
            pytest.skip(f"{path} not found")
        config = load_cv_config_with_object(path)
        assert config.model.type == "fcn"
        assert "mIoU" in config.eval.metrics

    def test_override_at_load_time(self):
        path = self.CONFIG_DIR / "classification" / "resnet50_imagenet.yaml"
        if not path.exists():
            pytest.skip(f"{path} not found")
        config = load_cv_config_with_object(path, overrides=["schedule.base_lr=0.5"])
        assert config.schedule.base_lr == 0.5

    def test_cross_validation(self):
        path = self.CONFIG_DIR / "classification" / "resnet50_imagenet.yaml"
        if not path.exists():
            pytest.skip(f"{path} not found")
        config = load_cv_config_with_object(path)
        errors = validate_cross_section(config)
        assert len(errors) == 0, f"unexpected validation errors: {errors}"


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_stage(self):
        with pytest.raises(ValueError, match="run.stage"):
            config = CVConfig.from_mapping({"run": {"stage": "invalid"}})
            config.validate()

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="engine.backend"):
            config = CVConfig.from_mapping({
                "run": {"stage": "train"},
                "model": {"type": "resnet"},
                "data": {"train": {"dataset_name": "ds"}},
                "engine": {"backend": "bad"},
            })
            config.validate()

    def test_missing_model_type_for_train(self):
        with pytest.raises(ValueError, match="model.type"):
            config = CVConfig.from_mapping({
                "run": {"stage": "train"},
                "data": {"train": {"dataset_name": "ds"}},
            })
            config.validate()

    def test_missing_dataset_for_train(self):
        with pytest.raises(ValueError, match="dataset_name"):
            config = CVConfig.from_mapping({
                "run": {"stage": "train"},
                "model": {"type": "resnet"},
            })
            config.validate()


# ---------------------------------------------------------------------------
# CVConfig construction (direct, no YAML)
# ---------------------------------------------------------------------------


class TestCVConfigDirect:
    def test_minimal_train_config(self):
        config = CVConfig.from_mapping({
            "run": {"stage": "train"},
            "model": {"type": "resnet50"},
            "data": {"train": {"dataset_name": "ImageNet"}},
        })
        config.validate()
        assert config.run.stage == "train"
        assert config.model.type == "resnet50"

    def test_to_dict(self):
        config = CVConfig.from_mapping({
            "run": {"stage": "train", "name": "exp1"},
            "model": {"type": "resnet50"},
            "data": {"train": {"dataset_name": "ImageNet"}},
        })
        d = config.to_dict()
        assert d["run"]["name"] == "exp1"
        assert d["model"]["type"] == "resnet50"