"""Classification heads — registered via HEADS registry."""

from __future__ import annotations

import torch
import torch.nn as nn

from elysia_cv.core.registry import HEADS


@HEADS.register_module()
class LinearClsHead(nn.Module):
    """Simple linear classification head.

    Assumes input is already pooled (B, C).
    """

    def __init__(self, num_classes: int = 1000, in_channels: int = 2048, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_ratio = dropout

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


@HEADS.register_module()
class ClsHead(nn.Module):
    """General classification head (dropout + linear)."""

    def __init__(self, num_classes: int = 1000, in_channels: int = 2048,
                 hidden_dims: tuple[int, ...] = (),
                 dropout: float = 0.0, **kwargs) -> None:
        super().__init__()
        layers = []
        prev_dim = in_channels
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.flatten(1)
        return self.classifier(x)