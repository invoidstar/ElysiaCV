"""YOLO model plugin — MODEL_PLUGIN for auto‑discovery.

Registers YOLOv10/v11/v12/YOLO26 (n/s/m/l/x) as ElysiaCV detection models.
"""

from __future__ import annotations

from elysia_cv.core.api_version import CORE_API_VERSION
from elysia_cv.core.capabilities import MODEL_PLUGIN_CAPABILITIES
from elysia_cv.core.plugin import SimpleModelPlugin
from elysia_cv.core.readiness import PluginReadinessLevel
from elysia_cv.models.contract import ModelSpec

from .variants import get_variant, list_variants


def _make_specs() -> list[ModelSpec]:
    """Generate a ModelSpec for every registered YOLO variant."""
    specs: list[ModelSpec] = []
    for key in list_variants():
        v = get_variant(key)
        specs.append(
            ModelSpec(
                name=key,
                task_type="detection",
                backbone=f"CSPDarkNet-{v.block}",
                neck="PAN",
                head="DetectHead" if v.head == "standard" else "V10DetectHead",
                input_size=(640, 640),
                extra={
                    "version": v.version,
                    "depth": v.depth,
                    "width": v.width,
                },
            )
        )

    # HDMS‑YOLO variants (YOLOv11n/s/m with SimSPPF + EMA)
    for base in ("yolov11n", "yolov11s", "yolov11m"):
        v = get_variant(base)
        specs.append(
            ModelSpec(
                name=f"hdms_{base}",
                task_type="detection",
                backbone="CSPDarkNet-c3k2 + SimSPPF",
                neck="PAN + EMA",
                head="DetectHead",
                input_size=(640, 640),
                extra={
                    "version": "hdms",
                    "base_model": base,
                    "depth": v.depth,
                    "width": v.width,
                    "improvements": ["sim_sppf", "ema_attention"],
                },
            )
        )
    return specs


def get_model_plugin() -> SimpleModelPlugin:
    return SimpleModelPlugin(
        name="yolo",
        task_type="detection",
        api_version=CORE_API_VERSION,
        capabilities=MODEL_PLUGIN_CAPABILITIES,
        readiness_level=PluginReadinessLevel.trainable,
        model_specs=_make_specs(),
    )


MODEL_PLUGIN = get_model_plugin()