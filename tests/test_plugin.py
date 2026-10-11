"""Unit tests for plugin discovery and core modules."""

from __future__ import annotations

from elysia_cv.core.api_version import CORE_API_VERSION, api_major, is_compatible_api_version
from elysia_cv.core.errors import ElysiaCoreError, PluginApiVersionError, PluginDiscoveryError, PluginError
from elysia_cv.core.readiness import PluginReadinessLevel, normalize_readiness_level, readiness_at_least


class TestApiVersion:
    def test_core_api_version_is_string(self):
        assert isinstance(CORE_API_VERSION, str)
        assert CORE_API_VERSION == "1.0"

    def test_api_major(self):
        assert api_major("1.0") == 1
        assert api_major("2.5") == 2
        assert api_major("invalid") == -1

    def test_is_compatible(self):
        assert is_compatible_api_version("1.0")
        assert is_compatible_api_version("1.5")
        assert not is_compatible_api_version("2.0")
        assert not is_compatible_api_version("0.9")


class TestErrors:
    def test_hierarchy(self):
        assert issubclass(PluginError, ElysiaCoreError)
        assert issubclass(PluginApiVersionError, PluginError)
        assert issubclass(PluginDiscoveryError, PluginError)


class TestReadiness:
    def test_normalize_enum(self):
        assert normalize_readiness_level(PluginReadinessLevel.production_ready) == PluginReadinessLevel.production_ready

    def test_normalize_int(self):
        assert normalize_readiness_level(3) == PluginReadinessLevel.evaluation_ready

    def test_normalize_string(self):
        assert normalize_readiness_level("trainable") == PluginReadinessLevel.trainable
        assert normalize_readiness_level("production-ready") == PluginReadinessLevel.production_ready

    def test_normalize_default(self):
        assert normalize_readiness_level(None) == PluginReadinessLevel.metadata_only

    def test_at_least(self):
        assert readiness_at_least("trainable", "metadata_only")
        assert not readiness_at_least("metadata_only", "trainable")
        assert readiness_at_least("production_ready", "production_ready")


class TestPluginDiscovery:
    def test_discovers_dummy_plugin(self):
        from elysia_cv.core.plugin import discover_model_plugins
        plugins = discover_model_plugins()
        names = {p.plugin.name for p in plugins}
        assert "dummy" in names, f"Expected 'dummy' in {names}"

    def test_dummy_plugin_has_task_type(self):
        from elysia_cv.core.plugin import discover_model_plugins
        plugins = discover_model_plugins()
        dummy = next(p for p in plugins if p.plugin.name == "dummy")
        assert getattr(dummy.plugin, "task_type", None) == "classification"

    def test_model_registry(self):
        from elysia_cv.models.registry import ModelRegistry
        registry = ModelRegistry.discover()
        assert "dummy" in registry
        assert len(registry) >= 1

        classification_models = registry.list_models("classification")
        assert "dummy" in classification_models