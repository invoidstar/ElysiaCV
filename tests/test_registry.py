"""Unit tests for the Registry system."""

from __future__ import annotations

import pytest

from elysia_cv.core.registry import BACKBONES, NECKS, HEADS, TRANSFORMS, LOSSES, METRICS, HOOKS, Registry


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test")

        @reg.register_module()
        class Foo:
            pass

        assert "Foo" in reg
        assert reg.get("Foo") is Foo
        assert len(reg) == 1

    def test_register_with_name(self):
        reg = Registry("test")

        @reg.register_module(name="bar")
        class Foo2:
            pass

        assert "bar" in reg
        assert "Foo2" not in reg
        assert reg.get("bar") is Foo2

    def test_duplicate_raises(self):
        reg = Registry("test")

        @reg.register_module()
        class Baz:
            pass

        with pytest.raises(KeyError):
            @reg.register_module()
            class Baz:  # noqa: F811
                pass

    def test_force_overwrite(self):
        reg = Registry("test")

        @reg.register_module()
        class Qux:
            pass

        @reg.register_module(force=True)
        class Qux:  # noqa: F811
            x = 42

        assert reg.get("Qux").x == 42

    def test_missing_key_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_build_from_cfg(self):
        reg = Registry("test")

        @reg.register_module()
        class Adder:
            def __init__(self, a: int, b: int = 0):
                self.result = a + b

        obj = reg.build({"type": "Adder", "a": 3, "b": 5})
        assert obj.result == 8

    def test_build_missing_type(self):
        reg = Registry("test")
        with pytest.raises(ValueError, match="missing required key 'type'"):
            reg.build({"a": 1})

    def test_keys(self):
        reg = Registry("test")

        @reg.register_module()
        class A: pass

        @reg.register_module(name="custom")
        class B: pass

        assert set(reg.keys()) == {"A", "custom"}


class TestGlobalRegistries:
    def test_backbones_exists(self):
        assert BACKBONES.name == "backbone"
        # ResNet backbone auto-registered by elysia_cv._register_default_components()
        assert len(BACKBONES) >= 1

    def test_transforms_exists(self):
        assert TRANSFORMS.name == "transform"

    def test_losses_exists(self):
        assert LOSSES.name == "loss"

    def test_metrics_exists(self):
        assert METRICS.name == "metric"

    def test_hooks_exists(self):
        assert HOOKS.name == "hook"

    def test_build_with_overrides(self):
        reg = Registry("test")

        @reg.register_module()
        class Double:
            def __init__(self, x: int, y: int = 0):
                self.val = (x + y) * 2

        obj = reg.build({"type": "Double", "x": 10}, y=2)
        assert obj.val == 24