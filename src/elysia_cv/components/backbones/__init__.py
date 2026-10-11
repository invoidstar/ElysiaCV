"""Backbone components — auto-registered via BACKBONES registry.

Import sub-modules here to trigger ``@BACKBONES.register_module()`` decorators.
"""

from __future__ import annotations

from elysia_cv.components.backbones.resnet import ResNet  # noqa: F401
from elysia_cv.components.backbones.mlp import MLP  # noqa: F401
from elysia_cv.components.backbones.simple_cnn import SimpleCNN  # noqa: F401
from elysia_cv.components.backbones.vit import ViT  # noqa: F401