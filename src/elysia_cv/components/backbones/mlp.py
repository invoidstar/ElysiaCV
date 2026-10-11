"""MLP backbone — flatten + FC layers, registered via BACKBONES registry.

A simple multi-layer perceptron baseline for image classification.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from elysia_cv.core.registry import BACKBONES


@BACKBONES.register_module()
class MLP(nn.Module):
    """MLP backbone for image classification.

    Flattens the input image and passes through a stack of
    Linear → BatchNorm1d → GELU blocks, ending with a linear
    projection to ``hidden_dim``.

    Args:
        input_size: (H, W) of input image.
        in_channels: Input channels (3 for RGB).
        hidden_dims: List of hidden layer widths.
        dropout: Dropout rate after each hidden layer.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (448, 448),
        in_channels: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        h, w = input_size
        self._input_size = input_size
        self._in_channels = in_channels
        self.pool_last = True  # compatible with cls head

        if hidden_dims is None:
            hidden_dims = [4096, 2048, 1024]

        in_dim = in_channels * h * w
        layers: list[nn.Module] = []

        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden

        self._hidden_dim = hidden_dims[-1] if hidden_dims else in_dim
        self.layers = nn.Sequential(*layers)

    @property
    def out_channels(self) -> list[int]:
        return [self._hidden_dim]

    def forward(self, x: Any) -> Any:
        x = x.flatten(1)  # (B, C, H, W) → (B, C*H*W)
        return self.layers(x)