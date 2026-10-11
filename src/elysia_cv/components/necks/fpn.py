"""Feature Pyramid Network (FPN) — registered via NECKS registry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import NECKS


@NECKS.register_module()
class FPN(nn.Module):
    """Feature Pyramid Network.

    Takes multi-scale backbone features and produces a unified set of
    feature maps at stride {4, 8, 16, 32, 64} for use by detection heads.

    Args:
        in_channels: Channels of input feature maps (e.g. [256, 512, 1024, 2048]).
        out_channels: Output channels for all pyramid levels.
        num_outs: Number of output pyramid levels (default 5).
        start_level: Index of first input level to use (default 0).
    """

    def __init__(
        self,
        in_channels: list[int] | None = None,
        out_channels: int = 256,
        num_outs: int = 5,
        start_level: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level

        in_channels = in_channels or [256, 512, 1024, 2048]
        self.in_channels = in_channels
        self.num_ins = len(in_channels)

        # Lateral convolutions (1×1 to match out_channels)
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            )

        # Output convolutions (3×3 on merged features)
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        # Extra levels (P6 from P5 via stride-2 pooling, P7 for larger anchors)
        self.extra_convs = nn.ModuleList()
        if num_outs > self.num_ins:
            for i in range(self.num_ins, num_outs):
                if i == self.num_ins:
                    # P6: stride-2 on last output
                    self.extra_convs.append(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                    )
                else:
                    # P7+: stride-2 on previous extra
                    self.extra_convs.append(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                    )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Build feature pyramid from multi-scale inputs.

        Args:
            inputs: List of feature tensors from backbone (bottom-up).

        Returns:
            list of pyramid feature maps (top-down).
        """
        assert len(inputs) == self.num_ins, f"expected {self.num_ins} inputs, got {len(inputs)}"

        # Top-down pathway
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]

        # Build from top to bottom
        for i in range(self.num_ins - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="nearest"
            )

        # Output convs
        outs = [conv(laterals[i]) for i, conv in enumerate(self.fpn_convs)]

        # Extra levels
        if self.num_outs > self.num_ins:
            for i, extra in enumerate(self.extra_convs):
                outs.append(extra(outs[-1]))

        return outs


@NECKS.register_module()
class IdentityNeck(nn.Module):
    """Identity neck — passes features through unchanged.

    For architectures that don't need a neck (e.g. FCN segmentation).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        return x