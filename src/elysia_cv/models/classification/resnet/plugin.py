"""ResNet model plugin — exposes MODEL_PLUGIN for auto-discovery."""

from __future__ import annotations

from elysia_cv.core.api_version import CORE_API_VERSION
from elysia_cv.core.capabilities import MODEL_PLUGIN_CAPABILITIES
from elysia_cv.core.plugin import SimpleModelPlugin
from elysia_cv.core.readiness import PluginReadinessLevel
from elysia_cv.models.contract import ModelSpec


def get_model_plugin() -> SimpleModelPlugin:
    """Return the ResNet family plugin."""
    return SimpleModelPlugin(
        name="resnet",
        task_type="classification",
        api_version=CORE_API_VERSION,
        capabilities=MODEL_PLUGIN_CAPABILITIES,
        readiness_level=PluginReadinessLevel.trainable,
        model_specs=[
            ModelSpec(
                name="resnet18",
                task_type="classification",
                backbone="ResNet",
                head="LinearClsHead",
                input_size=(224, 224),
                num_params=11689512,
                extra={"depth": 18},
            ),
            ModelSpec(
                name="resnet34",
                task_type="classification",
                backbone="ResNet",
                head="LinearClsHead",
                input_size=(224, 224),
                num_params=21797672,
                extra={"depth": 34},
            ),
            ModelSpec(
                name="resnet50",
                task_type="classification",
                backbone="ResNet",
                head="LinearClsHead",
                input_size=(224, 224),
                num_params=25557032,
                extra={"depth": 50},
            ),
            ModelSpec(
                name="resnet101",
                task_type="classification",
                backbone="ResNet",
                head="LinearClsHead",
                input_size=(224, 224),
                num_params=44549160,
                extra={"depth": 101},
            ),
        ],
    )


# Module-level export for auto-discovery
MODEL_PLUGIN = get_model_plugin()