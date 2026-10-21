"""SlowFast — dual-pathway 3D ConvNet for action recognition.

Two parallel pathways:
  - **Slow pathway**: low frame rate (T_slow), high spatial res, captures spatial semantics.
  - **Fast pathway**: high frame rate (α × T_slow), low spatial res (β fewer channels),
    captures fine temporal motion.

Lateral connections fuse fast features into the slow pathway at each stage.

Reference: "SlowFast Networks for Video Recognition" (ICCV 2019)

Parameters:
    - α (alpha): frame rate ratio between fast and slow pathways (default 8).
    - β (beta):  channel ratio between fast and slow pathways (default 1/8).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv3d(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 0, bias: bool = False) -> nn.Conv3d:
    return nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)


def _bn3d(c: int) -> nn.BatchNorm3d:
    return nn.BatchNorm3d(c)


class _Bottleneck3D(nn.Module):
    """3D bottleneck block used in both pathways.

    Expansion factor is fixed at 4 (standard ResNet).
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.conv1 = _conv3d(inplanes, planes, k=1, bias=False)
        self.bn1 = _bn3d(planes)

        self.conv2 = _conv3d(
            planes, planes, k=3,
            s=(temporal_stride, spatial_stride, spatial_stride),
            p=1, bias=False,
        )
        self.bn2 = _bn3d(planes)

        self.conv3 = _conv3d(planes, planes * self.expansion, k=1, bias=False)
        self.bn3 = _bn3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride

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


class _ResStage(nn.Module):
    """One stage of ResNet blocks for a single pathway."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        blocks: int,
        spatial_stride: int = 1,
        temporal_stride: int = 1,
    ) -> None:
        super().__init__()
        expansion = _Bottleneck3D.expansion

        downsample = None
        if spatial_stride != 1 or temporal_stride != 1 or inplanes != planes * expansion:
            downsample = nn.Sequential(
                _conv3d(inplanes, planes * expansion, k=1,
                        s=(temporal_stride, spatial_stride, spatial_stride), bias=False),
                _bn3d(planes * expansion),
            )

        layers = [
            _Bottleneck3D(inplanes, planes, spatial_stride, temporal_stride, downsample)
        ]
        for _ in range(1, blocks):
            layers.append(_Bottleneck3D(planes * expansion, planes))

        self.layers = nn.Sequential(*layers)
        self.out_channels = planes * expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SlowFast(nn.Module):
    """SlowFast video classification model.

    Args:
        num_classes: Number of action classes.
        alpha:       Frame-rate ratio fast/slow (default 8).
        beta:        Channel ratio slow/fast (default 1/8).
        dropout:     Dropout rate before FC.
        blocks:      Number of bottleneck blocks per stage (default ResNet-50: [3,4,6,3]).
    """

    def __init__(
        self,
        num_classes: int = 101,
        alpha: int = 8,
        beta_inv: int = 8,          # 1/beta
        dropout: float = 0.5,
        blocks: tuple[int, ...] = (3, 4, 6, 3),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta_inv = beta_inv

        # ── Slow pathway stem ──
        # Slow: C=3, T stride=1, spatial stride=4, output w0
        w0 = 64
        self.slow_conv1 = nn.Conv3d(3, w0, kernel_size=(1, 7, 7),
                                    stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = _bn3d(w0)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1))

        # ── Fast pathway stem ──
        # Fast: C=3, T stride=alpha, fewer channels
        w0_fast = max(1, w0 // beta_inv)
        self.fast_conv1 = nn.Conv3d(3, w0_fast,
                                    kernel_size=(5, 7, 7),
                                    stride=(1, 2, 2),
                                    padding=(2, 3, 3), bias=False)
        self.fast_bn1 = _bn3d(w0_fast)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1))

        # ── Stages with lateral connections ──
        w1, w2, w3, w4 = 64, 128, 256, 512
        channels = [w1, w2, w3, w4]

        prev_slow = w0
        prev_fast = w0_fast

        self.stages = nn.ModuleList()
        self.lateral = nn.ModuleList()

        for idx, (planes, num_blocks) in enumerate(zip(channels, blocks)):
            spatial_stride = 2 if idx > 0 else 1
            # Slow pathway: no temporal stride after stage 1
            t_stride_slow = 2 if idx == 0 else 1

            s_stage = _ResStage(prev_slow, planes, num_blocks, spatial_stride, t_stride_slow)
            self.stages.append(s_stage)

            # Fast pathway: always temporal stride=alpha (except temporal-pool at stem)
            fast_planes = max(1, planes // beta_inv)
            t_stride_fast = 2 if idx == 0 else 1
            f_stage = _ResStage(prev_fast, fast_planes, num_blocks,
                                spatial_stride, t_stride_fast)
            self.stages.append(f_stage)

            # Lateral connection: fast → slow
            # Temporal conv with stride=alpha reduces fast frames to match slow
            slow_out = s_stage.out_channels
            fast_out = f_stage.out_channels
            lateral = nn.Sequential(
                nn.Conv3d(fast_out, slow_out,
                          kernel_size=(5, 1, 1),
                          stride=(alpha, 1, 1),
                          padding=(2, 0, 0), bias=False),
                _bn3d(slow_out),
                nn.ReLU(inplace=True),
            )
            self.lateral.append(lateral)

            prev_slow = slow_out
            prev_fast = fast_out

        # ── Head ──
        self.slow_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fast_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        total_dim = prev_slow + prev_fast

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(total_dim, num_classes)

        self.relu = nn.ReLU(inplace=True)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_slow: torch.Tensor,
                x_fast: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x_slow: Slow pathway input ``[B, C, T_slow, H, W]``.
                    If x_fast is None, fast input is derived by temporal downsampling.
            x_fast: Fast pathway input ``[B, C, T_fast, H, W]`` (T_fast = α × T_slow).

        Returns:
            Dict with ``"logits"`` key.
        """
        # If only one input is given, derive fast from slow
        if x_fast is None:
            # Temporal sampling: take every frame for fast pathway
            # This is simplified — ideally you'd sample at original frame rate
            T = x_slow.shape[2]
            # For simplicity, repeat slow frames α times (not ideal but functional)
            indices = torch.arange(0, T, step=max(1, T // (T * self.alpha)))[:T * self.alpha]
            indices = torch.clamp(indices, 0, T - 1)
            x_fast = x_slow[:, :, indices, :, :]

        # Slow pathway
        s = self.slow_conv1(x_slow)
        s = self.slow_bn1(s)
        s = self.relu(s)
        s = self.slow_maxpool(s)

        # Fast pathway
        f = self.fast_conv1(x_fast)
        f = self.fast_bn1(f)
        f = self.relu(f)
        f = self.fast_maxpool(f)

        # Stages with lateral connections
        for i in range(len(self.stages) // 2):
            s = self.stages[2 * i](s)          # slow stage
            f = self.stages[2 * i + 1](f)      # fast stage
            # Lateral: fuse fast→slow
            lat = self.lateral[i](f)
            s = s + lat

        # Global pool
        s = self.slow_avgpool(s).flatten(1)
        f = self.fast_avgpool(f).flatten(1)

        # Concat + classify
        feat = torch.cat([s, f], dim=1)
        feat = self.dropout(feat)
        logits = self.fc(feat)

        return {"logits": logits}