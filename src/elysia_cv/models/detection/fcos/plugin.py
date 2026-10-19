"""FCOS model plugin — MODEL_PLUGIN for auto-discovery.

FCOS = Fully Convolutional One-Stage object detector (anchor-free).
Uses center-sampling target assignment and per-FPN-level prediction.
"""

from __future__ import annotations

from elysia_cv.core.api_version import CORE_API_VERSION
from elysia_cv.core.capabilities import MODEL_PLUGIN_CAPABILITIES
from elysia_cv.core.plugin import SimpleModelPlugin
from elysia_cv.core.readiness import PluginReadinessLevel
from elysia_cv.models.contract import ModelSpec


def get_model_plugin() -> SimpleModelPlugin:
    return SimpleModelPlugin(
        name="fcos",
        task_type="detection",
        api_version=CORE_API_VERSION,
        capabilities=MODEL_PLUGIN_CAPABILITIES,
        readiness_level=PluginReadinessLevel.trainable,
        model_specs=[
            ModelSpec(
                name="fcos_r50",
                task_type="detection",
                backbone="ResNet",
                neck="FPN",
                head="DetHead",
                input_size=(1333, 800),
                extra={"backbone_depth": 50},
            ),
            ModelSpec(
                name="fcos_r101",
                task_type="detection",
                backbone="ResNet",
                neck="FPN",
                head="DetHead",
                input_size=(1333, 800),
                extra={"backbone_depth": 101},
            ),
        ],
    )


MODEL_PLUGIN = get_model_plugin()