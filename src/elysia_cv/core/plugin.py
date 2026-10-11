"""Lightweight plugin protocol and discovery — adapted from ElysiaRobot.

The goal: new models are added through plugin directories, not by
modifying a central registry. This module depends ONLY on the Python
standard library and core lightweight types so that read-only workflows
(doctor / dry-run / preflight) never trigger heavy model imports.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from elysia_cv.core.api_version import CORE_API_VERSION, is_compatible_api_version
from elysia_cv.core.capabilities import (
    DATASET_PLUGIN_CAPABILITIES,
    METRIC_PLUGIN_CAPABILITIES,
    MODEL_PLUGIN_CAPABILITIES,
)
from elysia_cv.core.errors import PluginApiVersionError, PluginDiscoveryError, PluginError
from elysia_cv.core.readiness import PluginReadinessLevel, normalize_readiness_level

# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------


class ModelPlugin(Protocol):
    """Model plugin protocol — the CV equivalent of ElysiaRobot's PolicyPlugin.

    Each plugin module MUST expose ``MODEL_PLUGIN`` or ``get_model_plugin()``
    returning an object that satisfies this protocol.  The Core only reads
    lightweight metadata; model construction, runtime, and loss are handled
    by the task-level runtime registry.
    """

    name: str
    api_version: str
    capabilities: Sequence[str]
    readiness_level: str
    task_type: str  # "classification" | "detection" | "segmentation" | "recognition"

    def model_specs(self) -> Iterable[Any]:
        """Return the ModelSpec list provided by this plugin."""


class DatasetPlugin(Protocol):
    """Dataset plugin protocol.

    Plugin modules MUST expose ``DATASET_PLUGIN`` or ``get_dataset_plugin()``.
    The Core only discovers lightweight metadata; data loading, transforms,
    and batching are handled by the data layer.
    """

    name: str
    api_version: str
    capabilities: Sequence[str]
    readiness_level: str

    def dataset_specs(self) -> Iterable[Mapping[str, Any]]:
        """Return data-source metadata provided by this plugin."""


class MetricPlugin(Protocol):
    """Metric plugin protocol.

    Plugin modules MUST expose ``METRIC_PLUGIN`` or ``get_metric_plugin()``.
    """

    name: str
    api_version: str
    capabilities: Sequence[str]
    readiness_level: str

    def metric_specs(self) -> Iterable[Mapping[str, Any]]:
        """Return metric metadata provided by this plugin."""


# ---------------------------------------------------------------------------
# Simple (data-class based) implementations
# ---------------------------------------------------------------------------


class SimpleModelPlugin:
    """Minimal model plugin implementation for built-in adapters and tests."""

    def __init__(
        self,
        *,
        name: str,
        task_type: str,
        model_specs: Iterable[Any],
        api_version: str = CORE_API_VERSION,
        capabilities: Sequence[str] = MODEL_PLUGIN_CAPABILITIES,
        readiness_level: str | PluginReadinessLevel = PluginReadinessLevel.metadata_only,
    ) -> None:
        if not str(name).strip():
            raise PluginError("model plugin name cannot be empty")
        self.name = str(name)
        self.task_type = str(task_type)
        self._model_specs = tuple(model_specs)
        self.api_version = str(api_version)
        self.capabilities = tuple(capabilities)
        self.readiness_level = normalize_readiness_level(readiness_level).name

    def model_specs(self) -> Iterable[Any]:
        return self._model_specs


class SimpleDatasetPlugin:
    """Minimal dataset plugin implementation."""

    def __init__(
        self,
        *,
        name: str,
        dataset_specs: Iterable[Mapping[str, Any]],
        api_version: str = CORE_API_VERSION,
        capabilities: Sequence[str] = DATASET_PLUGIN_CAPABILITIES,
        readiness_level: str | PluginReadinessLevel = PluginReadinessLevel.metadata_only,
    ) -> None:
        if not str(name).strip():
            raise PluginError("dataset plugin name cannot be empty")
        self.name = str(name)
        self._dataset_specs = tuple(dict(spec) for spec in dataset_specs)
        self.api_version = str(api_version)
        self.capabilities = tuple(capabilities)
        self.readiness_level = normalize_readiness_level(readiness_level).name

    def dataset_specs(self) -> Iterable[Mapping[str, Any]]:
        return self._dataset_specs


class SimpleMetricPlugin:
    """Minimal metric plugin implementation."""

    def __init__(
        self,
        *,
        name: str,
        metric_specs: Iterable[Mapping[str, Any]],
        api_version: str = CORE_API_VERSION,
        capabilities: Sequence[str] = METRIC_PLUGIN_CAPABILITIES,
        readiness_level: str | PluginReadinessLevel = PluginReadinessLevel.metadata_only,
    ) -> None:
        if not str(name).strip():
            raise PluginError("metric plugin name cannot be empty")
        self.name = str(name)
        self._metric_specs = tuple(dict(spec) for spec in metric_specs)
        self.api_version = str(api_version)
        self.capabilities = tuple(capabilities)
        self.readiness_level = normalize_readiness_level(readiness_level).name

    def metric_specs(self) -> Iterable[Mapping[str, Any]]:
        return self._metric_specs


# ---------------------------------------------------------------------------
# Loaded-wrapper dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModelPlugin:
    """A model plugin together with its source module name."""
    module_name: str
    plugin: ModelPlugin


@dataclass(frozen=True)
class LoadedDatasetPlugin:
    """A dataset plugin together with its source module name."""
    module_name: str
    plugin: DatasetPlugin


@dataclass(frozen=True)
class LoadedMetricPlugin:
    """A metric plugin together with its source module name."""
    module_name: str
    plugin: MetricPlugin


# ---------------------------------------------------------------------------
# Discovery infrastructure
# ---------------------------------------------------------------------------


def discover_model_plugins(
    *,
    root_package: str = "elysia_cv.models",
    extra_modules: Sequence[str] = (),
    env_var: str = "ELYSIA_MODEL_PLUGIN_MODULES",
) -> tuple[LoadedModelPlugin, ...]:
    """Discover built-in and external model plugins.

    Discovery order:
    1. Scan ``<root_package>/*/plugin.py`` directories;
    2. Load explicit *extra_modules*;
    3. Load comma-separated modules from the environment variable.
    """
    module_names = [
        *_iter_builtin_model_plugin_modules(root_package),
        *extra_modules,
        *_env_plugin_modules(env_var),
    ]
    loaded: list[LoadedModelPlugin] = []
    seen: set[str] = set()
    for module_name in module_names:
        if module_name in seen:
            continue
        seen.add(module_name)
        loaded.append(load_model_plugin(module_name))
    return tuple(loaded)


def discover_dataset_plugins(
    *,
    root_package: str = "elysia_cv.data",
    extra_modules: Sequence[str] = (),
    env_var: str = "ELYSIA_DATASET_PLUGIN_MODULES",
) -> tuple[LoadedDatasetPlugin, ...]:
    """Discover built-in and external dataset plugins."""
    module_names = [
        *_iter_builtin_plugin_modules(root_package, suffix="_plugin", package_leaf="plugin"),
        *_iter_nested_plugin_modules(root_package, "plugins"),
        *extra_modules,
        *_env_plugin_modules(env_var),
    ]
    loaded: list[LoadedDatasetPlugin] = []
    seen: set[str] = set()
    for module_name in module_names:
        if module_name in seen:
            continue
        seen.add(module_name)
        loaded.append(load_dataset_plugin(module_name))
    return tuple(loaded)


def discover_metric_plugins(
    *,
    root_package: str = "elysia_cv.metrics",
    extra_modules: Sequence[str] = (),
    env_var: str = "ELYSIA_METRIC_PLUGIN_MODULES",
) -> tuple[LoadedMetricPlugin, ...]:
    """Discover built-in and external metric plugins."""
    module_names = [
        *_iter_builtin_plugin_modules(root_package, suffix="_metric", package_leaf="plugin"),
        *extra_modules,
        *_env_plugin_modules(env_var),
    ]
    loaded: list[LoadedMetricPlugin] = []
    seen: set[str] = set()
    for module_name in module_names:
        if module_name in seen:
            continue
        seen.add(module_name)
        loaded.append(load_metric_plugin(module_name))
    return tuple(loaded)


# ---------------------------------------------------------------------------
# Per-plugin loaders
# ---------------------------------------------------------------------------


def load_model_plugin(module_name: str) -> LoadedModelPlugin:
    """Load a single model plugin module and validate API version."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise PluginDiscoveryError(
            f"failed to import model plugin module {module_name!r}: {exc}"
        ) from exc

    plugin = _model_plugin_from_module(module, module_name)
    if not is_compatible_api_version(plugin.api_version):
        raise PluginApiVersionError(
            f"model plugin {plugin.name!r} uses api_version={plugin.api_version!r}, "
            f"but core expects {CORE_API_VERSION!r}"
        )
    return LoadedModelPlugin(module_name=module_name, plugin=plugin)


def load_dataset_plugin(module_name: str) -> LoadedDatasetPlugin:
    """Load a single dataset plugin module and validate API version."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise PluginDiscoveryError(
            f"failed to import dataset plugin module {module_name!r}: {exc}"
        ) from exc

    plugin = _dataset_plugin_from_module(module, module_name)
    if not is_compatible_api_version(plugin.api_version):
        raise PluginApiVersionError(
            f"dataset plugin {plugin.name!r} uses api_version={plugin.api_version!r}, "
            f"but core expects {CORE_API_VERSION!r}"
        )
    return LoadedDatasetPlugin(module_name=module_name, plugin=plugin)


def load_metric_plugin(module_name: str) -> LoadedMetricPlugin:
    """Load a single metric plugin module and validate API version."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise PluginDiscoveryError(
            f"failed to import metric plugin module {module_name!r}: {exc}"
        ) from exc

    plugin = _metric_plugin_from_module(module, module_name)
    if not is_compatible_api_version(plugin.api_version):
        raise PluginApiVersionError(
            f"metric plugin {plugin.name!r} uses api_version={plugin.api_version!r}, "
            f"but core expects {CORE_API_VERSION!r}"
        )
    return LoadedMetricPlugin(module_name=module_name, plugin=plugin)


# ---------------------------------------------------------------------------
# Internal: extract plugin object from a module
# ---------------------------------------------------------------------------


def _model_plugin_from_module(module: Any, module_name: str) -> ModelPlugin:
    factory = getattr(module, "get_model_plugin", None)
    if callable(factory):
        plugin = factory()
    else:
        plugin = getattr(module, "MODEL_PLUGIN", None)
    if plugin is None:
        raise PluginDiscoveryError(
            f"model plugin module {module_name!r} must define MODEL_PLUGIN or get_model_plugin()"
        )
    _validate_model_plugin(plugin, module_name)
    return plugin


def _dataset_plugin_from_module(module: Any, module_name: str) -> DatasetPlugin:
    factory = getattr(module, "get_dataset_plugin", None)
    if callable(factory):
        plugin = factory()
    else:
        plugin = getattr(module, "DATASET_PLUGIN", None)
    if plugin is None:
        raise PluginDiscoveryError(
            f"dataset plugin module {module_name!r} must define DATASET_PLUGIN or get_dataset_plugin()"
        )
    _validate_dataset_plugin(plugin, module_name)
    return plugin


def _metric_plugin_from_module(module: Any, module_name: str) -> MetricPlugin:
    factory = getattr(module, "get_metric_plugin", None)
    if callable(factory):
        plugin = factory()
    else:
        plugin = getattr(module, "METRIC_PLUGIN", None)
    if plugin is None:
        raise PluginDiscoveryError(
            f"metric plugin module {module_name!r} must define METRIC_PLUGIN or get_metric_plugin()"
        )
    _validate_metric_plugin(plugin, module_name)
    return plugin


# ---------------------------------------------------------------------------
# Internal: validation helpers
# ---------------------------------------------------------------------------


def _validate_model_plugin(plugin: Any, module_name: str) -> None:
    for attr in ("name", "api_version", "capabilities", "task_type", "model_specs"):
        if not hasattr(plugin, attr):
            raise PluginDiscoveryError(
                f"model plugin module {module_name!r} missing required attr {attr!r}"
            )
    if not callable(plugin.model_specs):
        raise PluginDiscoveryError(
            f"model plugin module {module_name!r} model_specs must be callable"
        )


def _validate_dataset_plugin(plugin: Any, module_name: str) -> None:
    for attr in ("name", "api_version", "capabilities", "dataset_specs"):
        if not hasattr(plugin, attr):
            raise PluginDiscoveryError(
                f"dataset plugin module {module_name!r} missing required attr {attr!r}"
            )
    if not callable(plugin.dataset_specs):
        raise PluginDiscoveryError(
            f"dataset plugin module {module_name!r} dataset_specs must be callable"
        )


def _validate_metric_plugin(plugin: Any, module_name: str) -> None:
    for attr in ("name", "api_version", "capabilities", "metric_specs"):
        if not hasattr(plugin, attr):
            raise PluginDiscoveryError(
                f"metric plugin module {module_name!r} missing required attr {attr!r}"
            )
    if not callable(plugin.metric_specs):
        raise PluginDiscoveryError(
            f"metric plugin module {module_name!r} metric_specs must be callable"
        )


# ---------------------------------------------------------------------------
# Internal: module iteration helpers
# ---------------------------------------------------------------------------


def _iter_builtin_model_plugin_modules(root_package: str) -> tuple[str, ...]:
    """Scan the models/ tree for plugin.py modules.

    Supports two layouts:

    1. Task-grouped (standard):
       ``<root>/<group>/<name>/plugin.py``
       e.g. ``models/classification/resnet/plugin.py``

    2. Flat top-level (test/debug):
       ``<root>/<name>/plugin.py``
       e.g. ``models/dummy/plugin.py``

    Also scans ``<root>/<group>/community/<name>/plugin.py``.
    """
    try:
        package = importlib.import_module(root_package)
    except Exception as exc:
        raise PluginDiscoveryError(
            f"failed to import model plugin root package {root_package!r}: {exc}"
        ) from exc

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return ()

    modules: list[str] = []
    prefix = f"{root_package}."

    for group_info in pkgutil.iter_modules(package_path, prefix):
        if not group_info.ispkg:
            continue

        # --- Layout 2: flat top-level model ---
        if _package_has_leaf_module(group_info.name, "plugin"):
            modules.append(f"{group_info.name}.plugin")

        # --- Layout 1: task-grouped ---
        try:
            group_pkg = importlib.import_module(group_info.name)
        except Exception:
            continue
        group_path = getattr(group_pkg, "__path__", None)
        if group_path is None:
            continue
        group_prefix = f"{group_info.name}."
        for model_info in pkgutil.iter_modules(group_path, group_prefix):
            if model_info.ispkg and _package_has_leaf_module(model_info.name, "plugin"):
                modules.append(f"{model_info.name}.plugin")

            # Also scan community/<name>/plugin.py
            if model_info.name.endswith(".community"):
                try:
                    comm_pkg = importlib.import_module(model_info.name)
                except Exception:
                    continue
                comm_path = getattr(comm_pkg, "__path__", None)
                if comm_path is None:
                    continue
                comm_prefix = f"{model_info.name}."
                for comm_info in pkgutil.iter_modules(comm_path, comm_prefix):
                    if comm_info.ispkg and _package_has_leaf_module(comm_info.name, "plugin"):
                        modules.append(f"{comm_info.name}.plugin")

    return tuple(sorted(set(modules)))  # dedup in case a dir matches both layouts


def _iter_builtin_plugin_modules(
    root_package: str, *, suffix: str, package_leaf: str
) -> tuple[str, ...]:
    """Scan flat and nested plugin modules below *root_package*."""
    try:
        package = importlib.import_module(root_package)
    except Exception as exc:
        raise PluginDiscoveryError(
            f"failed to import plugin root package {root_package!r}: {exc}"
        ) from exc

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return ()

    modules: list[str] = []
    prefix = f"{root_package}."
    for info in pkgutil.iter_modules(package_path, prefix):
        short_name = info.name.rsplit(".", 1)[-1]
        if info.ispkg:
            candidate = f"{info.name}.{package_leaf}"
            if _package_has_leaf_module(info.name, package_leaf):
                modules.append(candidate)
        elif short_name.endswith(suffix):
            modules.append(info.name)
    return tuple(sorted(modules))


def _iter_nested_plugin_modules(root_package: str, nested_package: str) -> tuple[str, ...]:
    """Scan ``<root>.<nested>/<name>/plugin.py``."""
    try:
        package = importlib.import_module(f"{root_package}.{nested_package}")
    except Exception:
        return ()
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return ()
    modules: list[str] = []
    prefix = f"{root_package}.{nested_package}."
    for info in pkgutil.iter_modules(package_path, prefix):
        if info.ispkg and _package_has_leaf_module(info.name, "plugin"):
            modules.append(f"{info.name}.plugin")
    return tuple(sorted(modules))


def _package_has_leaf_module(package_name: str, leaf_module: str) -> bool:
    """Check whether *package_name* has a sub-module named *leaf_module*."""
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return False
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return False
    return any(
        info.name.rsplit(".", 1)[-1] == leaf_module for info in pkgutil.iter_modules(package_path)
    )


def _env_plugin_modules(env_var: str) -> tuple[str, ...]:
    value = os.environ.get(env_var, "")
    return tuple(part.strip() for part in value.split(",") if part.strip())