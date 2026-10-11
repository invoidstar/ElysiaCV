"""Simple CNN backbone — lightweight ConvNet baseline.

A minimal CNN without residual connections for fair comparison
against ResNet and ViT on fine-grained cassava classification.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from elysia_cv.core.registry import BACKBONES


@BACKBONES.register_module()
class SimpleCNN(nn.Module):
    """Lightweight ConvNet for image classification.

    5-stage Conv2d → BatchNorm2d → ReLU → MaxPool2d pipeline,
    ending with adaptive average pooling and a linear projection.

    Args:
        in_channels: Input channels (3 for RGB).
        base_channels: Starting channel count, doubled each stage.
        num_classes: Output dimensionality (embedding dim for head).
        dropout: Dropout after final projection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 512,
        dropout: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.pool_last = True

        c = base_channels
        self.stage1 = self._conv_block(in_channels, c)
        self.stage2 = self._conv_block(c, c * 2)
        self.stage3 = self._conv_block(c * 2, c * 4)
        self.stage4 = self._conv_block(c * 4, c * 8)
        self.stage5 = self._conv_block(c * 8, c * 8)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(c * 8, num_classes)
        self._out_dim = num_classes

    @staticmethod
    def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    @property
    def out_channels(self) -> list[int]:
        return [self._out_dim]

    def forward(self, x: Any) -> Any:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.proj(x)
        x = self.dropout(x)
        return x