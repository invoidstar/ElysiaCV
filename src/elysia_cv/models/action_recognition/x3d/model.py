"""X3D — eXpand 3D ConvNet for efficient action recognition.

X3D expands a 2D ResNet along multiple axes:
  - Temporal duration (T): number of input frames
  - Frame rate (τ): temporal stride of early layers
  - Spatial resolution (S): input size (typically 160 or 224)
  - Network depth (D): layers per stage
  - Channel width (W): channels per stage
  - Bottleneck ratio (B): expansion in bottleneck blocks

Reference: "X3D: Expanding Architectures for Efficient Video Recognition" (CVPR 2020)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv3d(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 0, bias: bool = False) -> nn.Conv3d:
    return nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)


def _bn3d(c: int) -> nn.BatchNorm3d:
    return nn.BatchNorm3d(c)


class _X3DBottleneck(nn.Module):
    """X3D bottleneck block with temporal striding support.

    Structure::
        1×1×1 conv (reduce) → 3×3×3 conv (spatial+temp) → 1×1×1 conv (expand)

    Args:
        inplanes:   Input channels.
        planes:     Output channels of the middle conv.
        expansion:  Bottleneck expansion factor (default 4 → 1×1 expand to 4× planes).
        stride:     Spatial stride of the 3×3×3 conv.
        temporal_stride: Temporal stride of the 3×3×3 conv.
        downsample: Optional projection shortcut (for strided blocks).
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        expansion: int = 4,
        stride: int = 1,
        temporal_stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        outplanes = planes * expansion

        self.conv1 = _conv3d(inplanes, planes, k=1, bias=False)
        self.bn1 = _bn3d(planes)

        self.conv2 = _conv3d(
            planes, planes, k=3,
            s=(temporal_stride, stride, stride),
            p=(1, 1, 1), bias=False,
        )
        self.bn2 = _bn3d(planes)

        self.conv3 = _conv3d(planes, outplanes, k=1, bias=False)
        self.bn3 = _bn3d(outplanes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class X3D(nn.Module):
    """X3D video classification model.

    Input: ``[B, C, T, H, W]`` (NCDHW format, as produced by FormatShape NCTHW).

    Variants and their key parameters:

        ======  ===  =====  ====  =====  =====  ======
        Name    T    τ     S     D     W     B
        ======  ===  =====  ====  =====  =====  ======
        XS      4    4     160   2.2x   2.2x   2.2x
        S       13   13    160   2.2x   2.2x   2.2x
        M       16   16    224   2.2x   2.2x   2.2x
        L       16   16    312   2.2x   5.0x   2.2x
        ======  ===  =====  ====  =====  =====  ======

    Args:
        variant:   One of ``"xs"``, ``"s"``, ``"m"``, ``"l"``.
        num_classes: Number of action classes.
        dropout:   Dropout rate before FC.
    """

    _CONFIGS = {
        "xs": {"T": 4,  "tau": 4,  "S": 160,   "D": 2.2, "W": 2.2, "B": 2.2},
        "s":  {"T": 13, "tau": 13, "S": 160,   "D": 2.2, "W": 2.2, "B": 2.2},
        "m":  {"T": 16, "tau": 16, "S": 224,   "D": 2.2, "W": 2.2, "B": 2.2},
        "l":  {"T": 16, "tau": 16, "S": 312,   "D": 2.2, "W": 5.0, "B": 2.2},
    }

    def __init__(
        self,
        variant: str = "xs",
        num_classes: int = 101,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in self._CONFIGS:
            raise ValueError(f"Unknown X3D variant: {variant!r}. Choose from: {list(self._CONFIGS.keys())}")

        cfg = self._CONFIGS[variant]
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Base channels (ResNet-18 stem: 64)
        base_w = int(64 * cfg["W"])
        w1, w2, w3, w4 = base_w, base_w * 2, base_w * 4, base_w * 8
        # Bottleneck expansion
        expansion = int(4 * cfg["B"])

        # Stem: 1×k×k conv → no temporal reduction yet
        # t_stride = 1 for stage 1, τ for later stages
        self.conv1 = nn.Conv3d(3, w1, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                               padding=(0, 3, 3), bias=False)
        self.bn1 = _bn3d(w1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        tau = cfg["tau"]
        self.layer1 = self._make_layer(w1, w1, int(3 * cfg["D"]), expansion,
                                       stride=1, t_stride=1)
        self.layer2 = self._make_layer(w1 * expansion, w2, int(4 * cfg["D"]), expansion,
                                       stride=2, t_stride=max(1, tau // 2))
        self.layer3 = self._make_layer(w2 * expansion, w3, int(6 * cfg["D"]), expansion,
                                       stride=2, t_stride=max(1, tau // 4))
        self.layer4 = self._make_layer(w3 * expansion, w4, int(3 * cfg["D"]), expansion,
                                       stride=2, t_stride=max(1, tau // 8))

        # Head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(w4 * expansion, num_classes)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        inplanes: int,
        planes: int,
        blocks: int,
        expansion: int,
        stride: int = 1,
        t_stride: int = 1,
    ) -> nn.Sequential:
        """Build a stage with multiple bottleneck blocks."""
        downsample = None
        if stride != 1 or t_stride != 1 or inplanes != planes * expansion:
            downsample = nn.Sequential(
                _conv3d(inplanes, planes * expansion, k=1,
                        s=(t_stride, stride, stride), bias=False),
                _bn3d(planes * expansion),
            )

        layers = [
            _X3DBottleneck(inplanes, planes, expansion, stride,
                           temporal_stride=t_stride, downsample=downsample)
        ]
        for _ in range(1, blocks):
            layers.append(
                _X3DBottleneck(planes * expansion, planes, expansion, stride=1)
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``[B, C, T, H, W]`` tensor.

        Returns:
            Dict with ``"logits"`` key, shape ``[B, num_classes]``.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)         # [B, C', 1, 1, 1]
        x = x.flatten(1)            # [B, C']
        x = self.dropout(x)
        logits = self.fc(x)

        return {"logits": logits}