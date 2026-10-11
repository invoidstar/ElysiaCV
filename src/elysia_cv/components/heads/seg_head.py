"""Segmentation heads — registered via HEADS registry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import HEADS


@HEADS.register_module()
class SegHead(nn.Module):
    """FCN-style segmentation head.

    Takes a single backbone feature map and upsamples to predict
    per-pixel class labels.

    Args:
        num_classes: Number of segmentation classes.
        in_channels: Input channels from backbone.
        in_index: Which backbone stage output to use (0-based).
        channels: Internal feature channels.
        num_convs: Number of conv layers before upsampling.
        dropout: Dropout ratio before output.
        align_corners: For bilinear interpolation.
    """

    def __init__(
        self,
        num_classes: int = 150,
        in_channels: int = 2048,
        in_index: int = 3,
        channels: int = 512,
        num_convs: int = 2,
        dropout: float = 0.1,
        align_corners: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.in_index = in_index
        self.align_corners = align_corners

        convs = []
        for i in range(num_convs):
            cin = in_channels if i == 0 else channels
            convs.append(nn.Conv2d(cin, channels, kernel_size=3, padding=1))
            convs.append(nn.BatchNorm2d(channels))
            convs.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*convs)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Forward segmentation head.

        Args:
            feats: Backbone features (single tensor or list from which
                   self.in_index is selected).

        Returns:
            Tensor [B, num_classes, H, W] logits.
        """
        if isinstance(feats, (list, tuple)):
            x = feats[self.in_index]
        else:
            x = feats

        x = self.convs(x)
        x = self.dropout(x)
        x = self.classifier(x)
        # Output: [B, C, H, W] — caller handles upsampling to input size
        return x