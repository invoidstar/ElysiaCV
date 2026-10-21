"""TSN model plugin — registers with ElysiaCV SimpleModelPlugin."""

from __future__ import annotations

from elysia_cv.core.plugin import SimpleModelPlugin

MODEL_PLUGIN = SimpleModelPlugin(
    name="tsn",
    task_type="action_recognition",
    model_specs=[
        {
            "type": "tsn",
            "backbone": "resnet50",
            "num_classes": 101,
            "description": "TSN with ResNet50 backbone",
        },
    ],
)