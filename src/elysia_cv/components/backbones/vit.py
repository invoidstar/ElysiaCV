"""Vision Transformer (ViT) backbone — registered via BACKBONES registry.

Wraps torchvision's ViT, stripping the classification head and exposing
the CLS token embedding for downstream classification.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from elysia_cv.core.registry import BACKBONES

# ── supported ViT variants ────────────────────────────────────────────
_VARIANT_TABLE: dict[str, dict[str, Any]] = {
    "vit_b_16": {
        "arch": "vit_b_16",
        "embed_dim": 768,
        "description": "ViT-Base/16 — 86M params, patch=16",
    },
    "vit_b_32": {
        "arch": "vit_b_32",
        "embed_dim": 768,
        "description": "ViT-Base/32 — 88M params, patch=32",
    },
    "vit_l_16": {
        "arch": "vit_l_16",
        "embed_dim": 1024,
        "description": "ViT-Large/16 — 307M params, patch=16",
    },
    "vit_l_32": {
        "arch": "vit_l_32",
        "embed_dim": 1024,
        "description": "ViT-Large/32 — 306M params, patch=32",
    },
    "vit_h_14": {
        "arch": "vit_h_14",
        "embed_dim": 1280,
        "description": "ViT-Huge/14 — 632M params, patch=14",
    },
}


@BACKBONES.register_module()
class ViT(nn.Module):
    """Vision Transformer backbone.

    Wraps ``torchvision.models.vision_transformer.VisionTransformer``,
    stripping the head so the CLS embedding is returned directly.

    Args:
        variant: One of ``vit_b_16``, ``vit_b_32``, ``vit_l_16``, ``vit_l_32``,
            ``vit_h_14``.
        image_size: Input image size (int).  Default 448 for fine-grained tasks.
        pretrained: Load ImageNet-21k / ImageNet-1k pretrained weights.
        frozen_stages: Number of encoder blocks to freeze (0 = none).
        dropout: Dropout rate (overrides default if > 0).
    """

    def __init__(
        self,
        variant: str = "vit_b_16",
        image_size: int = 448,
        pretrained: bool = False,
        frozen_stages: int = 0,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if variant not in _VARIANT_TABLE:
            raise ValueError(f"Unknown ViT variant: {variant!r}. Choose from {list(_VARIANT_TABLE)}.")

        info = _VARIANT_TABLE[variant]
        arch_name = info["arch"]
        self._embed_dim = info["embed_dim"]
        self.pool_last = True  # compatible with cls head

        import torchvision.models as tv_models

        weights = "DEFAULT" if pretrained else None

        factory = getattr(tv_models, arch_name, None)
        if factory is None:
            raise ImportError(f"torchvision has no model '{arch_name}'. Upgrade torchvision?")

        # Build ViT with a dummy head, then decompose to extract CLS token
        tv_vit = factory(
            weights=weights,
            image_size=image_size,
            num_classes=self._embed_dim,  # dummy, will be stripped
        )

        # Decompose ViT: conv_proj + encoder + class_token + pos_embedding
        self.conv_proj = tv_vit.conv_proj
        self.class_token = tv_vit.class_token
        self.encoder = tv_vit.encoder
        self._image_size = image_size

        # Register pos_embedding as a buffer
        if hasattr(tv_vit.encoder, "pos_embedding"):
            pos_embed = tv_vit.encoder.pos_embedding
        else:
            pos_embed = None

        self._pos_embedding = pos_embed

        # Dropout override
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze early encoder blocks
        if frozen_stages > 0:
            blocks = getattr(self.encoder, "layers", None) or getattr(self.encoder, "blocks", None) or []
            for idx in range(min(frozen_stages, len(blocks))):
                for param in blocks[idx].parameters():
                    param.requires_grad = False

    @property
    def out_channels(self) -> list[int]:
        return [self._embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Manual ViT forward, returning CLS token ``(B, embed_dim)``."""
        # Patch embedding
        x = self.conv_proj(x)  # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Prepend class token
        cls_tokens = self.class_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)

        # Positional encoding
        if self._pos_embedding is not None:
            pos_embed = self._pos_embedding
            if pos_embed.size(1) != x.size(1):
                # Interpolate if image_size differs from pretrained
                pos_embed = torch.nn.functional.interpolate(
                    pos_embed.transpose(1, 2).unsqueeze(0),
                    size=(x.size(1),),
                    mode="linear",
                ).squeeze(0).transpose(1, 2)
            x = x + pos_embed

        x = self._dropout(x)

        # Transformer encoder
        x = self.encoder(x)

        # CLS token
        cls = x[:, 0, :]  # (B, embed_dim)
        return cls