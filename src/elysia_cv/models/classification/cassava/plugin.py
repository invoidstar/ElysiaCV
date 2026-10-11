"""Cassava Leaf Disease model plugin — MODEL_PLUGIN for auto-discovery.

Fine-grained classification of 5 cassava leaf disease categories.
Covers 6 experimental architectures: MLP, SimpleCNN, ResNet18/34, ViT-B/16, ViT-L/16.
"""

from __future__ import annotations

from elysia_cv.core.api_version import CORE_API_VERSION
from elysia_cv.core.capabilities import MODEL_PLUGIN_CAPABILITIES
from elysia_cv.core.plugin import SimpleModelPlugin
from elysia_cv.core.readiness import PluginReadinessLevel
from elysia_cv.models.contract import ModelSpec


def get_model_plugin() -> SimpleModelPlugin:
    return SimpleModelPlugin(
        name="cassava",
        task_type="classification",
        api_version=CORE_API_VERSION,
        capabilities=MODEL_PLUGIN_CAPABILITIES,
        readiness_level=PluginReadinessLevel.trainable,
        model_specs=[
            # ── MLP baseline ──────────────────────────────────────
            ModelSpec(
                name="cassava_mlp",
                task_type="classification",
                backbone="MLP",
                head="LinearClsHead",
                input_size=(448, 448),
                extra={
                    "num_classes": 5,
                    "description": "MLP baseline — 4096→2048→1024, GELU + Dropout",
                    "params": "~7M",
                },
            ),
            # ── SimpleCNN baseline ────────────────────────────────
            ModelSpec(
                name="cassava_simple_cnn",
                task_type="classification",
                backbone="SimpleCNN",
                head="LinearClsHead",
                input_size=(448, 448),
                extra={
                    "num_classes": 5,
                    "description": "5-stage plain ConvNet — no residual connections",
                    "params": "~4M",
                    "head_in_channels": 512,
                },
            ),
            # ── ResNet family ─────────────────────────────────────
            ModelSpec(
                name="cassava_resnet18",
                task_type="classification",
                backbone="ResNet",
                head="LinearClsHead",
                input_size=(448, 448),
                extra={
                    "num_classes": 5,
                    "backbone_depth": 18,
                    "head_in_channels": 512,
                    "description": "ResNet-18 — 11M params",
                },
            ),
            ModelSpec(
                name="cassava_resnet34",
                task_type="classification",
                backbone="ResNet",
                head="LinearClsHead",
                input_size=(448, 448),
                extra={
                    "num_classes": 5,
                    "backbone_depth": 34,
                    "head_in_channels": 512,
                    "description": "ResNet-34 — 21M params",
                },
            ),
            ModelSpec(
                name="cassava_resnet50",
                task_type="classification",
                backbone="ResNet",
                head="LinearClsHead",
                input_size=(448, 448),
                extra={
                    "num_classes": 5,
                    "backbone_depth": 50,
                    "head_in_channels": 2048,
                    "description": "ResNet-50 — 25M params, strong baseline",
                },
            ),
            # ── ViT family ────────────────────────────────────────
            ModelSpec(
                name="cassava_vit_b16",
                task_type="classification",
                backbone="ViT",
                head="LinearClsHead",
                input_size=(448, 448),
                extra={
                    "num_classes": 5,
                    "vit_variant": "vit_b_16",
                    "head_in_channels": 768,
                    "description": "ViT-Base/16 — 86M params, patch=16",
                },
            ),
            ModelSpec(
                name="cassava_vit_l16",
                task_type="classification",
                backbone="ViT",
                head="LinearClsHead",
                input_size=(448, 448),
                extra={
                    "num_classes": 5,
                    "vit_variant": "vit_l_16",
                    "head_in_channels": 1024,
                    "description": "ViT-Large/16 — 307M params, patch=16",
                },
            ),
        ],
    )


MODEL_PLUGIN = get_model_plugin()