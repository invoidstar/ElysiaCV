"""FCN model plugin — MODEL_PLUGIN for auto-discovery."""

from __future__ import annotations

from elysia_cv.core.api_version import CORE_API_VERSION
from elysia_cv.core.capabilities import MODEL_PLUGIN_CAPABILITIES
from elysia_cv.core.plugin import SimpleModelPlugin
from elysia_cv.core.readiness import PluginReadinessLevel
from elysia_cv.models.contract import ModelSpec


def get_model_plugin() -> SimpleModelPlugin:
    return SimpleModelPlugin(
        name="fcn",
        task_type="segmentation",
        api_version=CORE_API_VERSION,
        capabilities=MODEL_PLUGIN_CAPABILITIES,
        readiness_level=PluginReadinessLevel.trainable,
        model_specs=[
            ModelSpec(
                name="fcn_r50",
                task_type="segmentation",
                backbone="ResNet",
                neck="IdentityNeck",
                head="SegHead",
                input_size=(2048, 512),
                extra={"backbone_depth": 50},
            ),
            ModelSpec(
                name="fcn_r101",
                task_type="segmentation",
                backbone="ResNet",
                neck="IdentityNeck",
                head="SegHead",
                input_size=(2048, 512),
                extra={"backbone_depth": 101},
            ),
        ],
    )


MODEL_PLUGIN = get_model_plugin()