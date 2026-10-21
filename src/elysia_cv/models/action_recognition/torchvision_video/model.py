"""Torchvision 3D CNN video models — R3D, MC3, R2Plus1D, S3D, MViTv2.

Wraps torchvision's built-in video classification models with a uniform
interface compatible with the ElysiaCV training loop.

Supported backbones (from ``torchvision.models.video``):
    - ``r3d_18``        — 3D ResNet-18 (33.2M, K400 pretrained)
    - ``mc3_18``        — Mixed Convolution 3D (11.5M, K400 pretrained)
    - ``r2plus1d_18``  — R(2+1)D ResNet-18 (31.4M, K400 pretrained)
    - ``s3d``           — Separable 3D CNN (K400 pretrained)

Input: ``[B, C, T, H, W]`` (5D tensor in NCDHW format).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import video as tv_video


_TV_VIDEO_MODELS = {
    "r3d_18": ("r3d_18", tv_video.R3D_18_Weights),
    "mc3_18": ("mc3_18", tv_video.MC3_18_Weights),
    "r2plus1d_18": ("r2plus1d_18", tv_video.R2Plus1D_18_Weights),
    "s3d": ("s3d", tv_video.S3D_Weights),
}


class TorchvisionVideo(nn.Module):
    """Thin wrapper around torchvision video models.

    Args:
        backbone_name: One of ``"r3d_18"``, ``"mc3_18"``, ``"r2plus1d_18"``, ``"s3d"``.
        num_classes:  Number of action classes.
        pretrained:   Load Kinetics-400 pretrained weights.
    """

    def __init__(
        self,
        backbone_name: str = "r3d_18",
        num_classes: int = 101,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name not in _TV_VIDEO_MODELS:
            raise ValueError(
                f"Unknown torchvision video backbone: {backbone_name!r}. "
                f"Choose from: {list(_TV_VIDEO_MODELS.keys())}"
            )

        model_name, weights_cls = _TV_VIDEO_MODELS[backbone_name]
        weights = weights_cls.KINETICS400_V1 if pretrained else None

        # torchvision models output [B, num_classes] directly
        self.model = tv_video.__dict__[model_name](
            weights=weights,
            num_classes=num_classes,
        )

        self._loss_fn: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``[B, C, T, H, W]`` tensor (NCDHW format).

        Returns:
            Dict with ``"logits"`` key, shape ``[B, num_classes]``.
        """
        logits = self.model(x)  # [B, num_classes]
        return {"logits": logits}