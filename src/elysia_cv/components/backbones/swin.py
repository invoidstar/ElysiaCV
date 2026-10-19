"""Swin Transformer backbone — registered via BACKBONES registry.

Supports ``features_only=True`` mode for multi-scale feature extraction,
which is required by HERBS and other FGVC necks (FPN, PIM).

Wraps ``timm.create_model`` with Swin Transformer variants.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from elysia_cv.core.registry import BACKBONES

# ── supported Swin variants ────────────────────────────────────────────
_VARIANT_TABLE: dict[str, dict[str, Any]] = {
    "swin_tiny": {
        "timm_name": "swin_tiny_patch4_window7_224",
        "depths": [2, 2, 6, 2],
        "embed_dim": 96,
        "out_channels": [96, 192, 384, 768],
        "description": "Swin-Tiny — 29M params",
    },
    "swin_small": {
        "timm_name": "swin_small_patch4_window7_224",
        "depths": [2, 2, 18, 2],
        "embed_dim": 96,
        "out_channels": [96, 192, 384, 768],
        "description": "Swin-Small — 50M params",
    },
    "swin_base": {
        "timm_name": "swin_base_patch4_window7_224",
        "depths": [2, 2, 18, 2],
        "embed_dim": 128,
        "out_channels": [128, 256, 512, 1024],
        "description": "Swin-Base — 88M params",
    },
    "swin_base_384": {
        "timm_name": "swin_base_patch4_window12_384",
        "depths": [2, 2, 18, 2],
        "embed_dim": 128,
        "out_channels": [128, 256, 512, 1024],
        "description": "Swin-Base @384 — 88M params, window=12",
    },
    "swin_large_384": {
        "timm_name": "swin_large_patch4_window12_384_in22k",
        "depths": [2, 2, 18, 2],
        "embed_dim": 192,
        "out_channels": [192, 384, 768, 1536],
        "description": "Swin-Large @384 IN22k — 197M params",
    },
}


@BACKBONES.register_module()
class SwinTransformer(nn.Module):
    """Swin Transformer backbone with multi-scale ``features_only`` output.

    Parameters
    ----------
    variant : str
        One of ``swin_tiny``, ``swin_small``, ``swin_base``, ``swin_base_384``,
        ``swin_large_384``.
    pretrained : bool
        Load pretrained weights from timm model zoo.
    image_size : int
        Input image size.  If ``None``, defaults to the variant's native size.
    features_only : bool
        If True, return a list of 4 multi-scale feature maps [stage1..stage4]
        in [B, S, C] format (sequence of tokens).  Required for HERBS / FPN necks.
    """

    def __init__(
        self,
        variant: str = "swin_base_384",
        pretrained: bool = True,
        image_size: int | None = None,
        features_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if variant not in _VARIANT_TABLE:
            raise ValueError(
                f"Unknown Swin variant: {variant!r}. Choose from {list(_VARIANT_TABLE)}."
            )

        info = _VARIANT_TABLE[variant]
        self._variant = variant
        self._out_channels = info["out_channels"]
        self._features_only = features_only
        self._image_size = image_size or (384 if "384" in variant else 224)

        import timm

        self.swin = timm.create_model(
            info["timm_name"],
            pretrained=pretrained,
            features_only=features_only,
            img_size=self._image_size,
            out_indices=(0, 1, 2, 3) if features_only else None,
        )

        # When features_only=True, timm wraps the model in FeatureListNet.
        # Store whether it's wrapped.
        self._is_feature_list = features_only

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    @property
    def pool_last(self) -> bool:
        return not self._features_only

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass.

        Returns:
            - If ``features_only=True``: list of 4 tensors [B, S_i, C_i]
              (stage1..stage4 token sequences).
            - Otherwise: single tensor [B, C_last] (pooled feature).
        """
        if self._features_only:
            feats = self.swin(x)  # FeatureListNet.forward() returns list
            # timm FeatureListNet returns NHWC [B, H, W, C] — convert to [B, H*W, C]
            out = []
            for f in feats:
                if len(f.shape) == 4:
                    B, H, W, C = f.shape  # NHWC from timm
                    f = f.reshape(B, H * W, C).contiguous()
                out.append(f)
            return out

        return self.swin.forward_features(x)