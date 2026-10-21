"""ST-Adapter: Spatio-Temporal Adapter for efficient video adaptation.

Freeze a pretrained image ViT backbone, then insert lightweight 3D conv
adapters between transformer blocks to capture spatio-temporal features.

Reference: "AIM: Adapting Image Models for Efficient Video Action Recognition"
(ICLR 2023, Yang et al.)

Structure::

    patch_embed → [Block → ST-Adapter → Block → ST-Adapter → ...] → norm → pool → FC
                     ↑ frozen ViT ↑                      ↑ trainable adapters ↑
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class _STAdapterBlock(nn.Module):
    """Spatio-Temporal adapter: 3D conv bottleneck inserted after a ViT block.

    ``Linear_down → 3D Conv(1×3×3) → Linear_up``

    Args:
        embed_dim:   Feature dimension of ViT.
        adapter_ratio: Reduction ratio for bottleneck (default 4).
        temporal_kernel: Temporal conv kernel (default 1 → no temporal modeling).
    """

    def __init__(
        self,
        embed_dim: int,
        adapter_ratio: int = 4,
        temporal_kernel: int = 1,
    ) -> None:
        super().__init__()
        hidden_dim = embed_dim // adapter_ratio

        self.down = nn.Linear(embed_dim, hidden_dim)
        self.conv3d = nn.Conv3d(
            hidden_dim, hidden_dim,
            kernel_size=(temporal_kernel, 3, 3),
            stride=1,
            padding=(temporal_kernel // 2, 1, 1),
        )
        self.up = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, H: int, W: int, T: int) -> torch.Tensor:
        """Apply 3D conv adapter.

        Args:
            x: ``[B*T, N+1, D]`` — token sequence (CLS + patches).
            H, W: Spatial resolution of feature map.
            T:    Number of temporal frames.

        Returns:
            ``[B*T, N+1, D]`` with residual adapter features added to patch
            tokens; CLS token passed through unchanged.
        """
        BT, Np1, D = x.shape
        cls_token = x[:, :1, :]           # [BT, 1, D]
        patch_tokens = x[:, 1:, :]        # [BT, N, D]

        # Reshape to [B*T, H, W, D]
        patch_spatial = patch_tokens.view(BT, H, W, D)
        B = BT // T

        # → [B, D, T, H, W] for 3D conv
        patch_3d = patch_spatial.view(B, T, H, W, D)
        patch_3d = patch_3d.permute(0, 4, 1, 2, 3)  # [B, D, T, H, W]

        # Adapter
        patch_3d = self.down(patch_3d.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)  # rough...
        # Actually let's do this right with proper reshape

        # Simpler approach: down-project, conv3d, up-project
        adapted = self.down(patch_tokens)              # [BT, N, hidden]
        adapted = adapted.view(BT, H, W, -1)           # [BT, H, W, hidden]
        adapted = adapted.view(B, T, H, W, -1)         # [B, T, H, W, hidden]
        adapted = adapted.permute(0, 4, 1, 2, 3)       # [B, hidden, T, H, W]

        adapted = self.conv3d(adapted)                  # [B, hidden, T, H, W]

        adapted = adapted.permute(0, 2, 3, 4, 1)       # [B, T, H, W, hidden]
        adapted = adapted.reshape(BT, H * W, -1)        # [BT, N, hidden]
        adapted = self.up(adapted)                      # [BT, N, D]

        # Residual connection
        adapted = self.norm(adapted)
        adapted = adapted + patch_tokens

        # Re-attach CLS token
        return torch.cat([cls_token, adapted], dim=1)


class STAdapter(nn.Module):
    """ST-Adapter video classification model.

    Args:
        backbone_name:  timm ViT backbone name (e.g. ``"vit_base_patch16_clip_224"``).
        num_classes:    Number of action classes.
        pretrained:     Load pretrained weights for backbone.
        adapter_ratio:  Bottleneck reduction ratio (default 4).
        dropout:        Dropout for classification head.
        freeze_backbone: Freeze ViT blocks (default True).
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_clip_224",
        num_classes: int = 101,
        pretrained: bool = True,
        adapter_ratio: int = 4,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.adapter_ratio = adapter_ratio

        # Load ViT backbone (strip classification head)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )

        embed_dim = self.backbone.embed_dim
        num_blocks = len(self.backbone.blocks)

        # Compute feature map size: assumes 224 × 224 input, patch_size=16 → 14×14
        patch_size = self.backbone.patch_embed.patch_size[0]
        self._spatial_size = 224 // patch_size  # 14

        # Freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Create adapters (one per block, skip last one)
        self.adapters = nn.ModuleList([
            _STAdapterBlock(embed_dim, adapter_ratio, temporal_kernel=3)
            for _ in range(num_blocks)
        ])

        # Classification head
        self.cls_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

        self.feat_dim = embed_dim

    def _parse_input(self, x: torch.Tensor):
        """Robustly parse video input into [B*T, C, H, W]."""
        shapes = list(x.shape)
        c_positions = [i for i, s in enumerate(shapes) if s == 3]

        if x.dim() == 4:
            if c_positions == [1] or c_positions == [2]:
                # [T, C, H, W] — common NCHW format
                x = x.view(shapes[0], 3, shapes[-2], shapes[-1])
            T_frames = x.shape[0]
            B = 1
        elif x.dim() == 5:
            if 2 in c_positions:
                # [B, T, C, H, W] — DataLoader stacked NCHW
                B, T_frames = shapes[0], shapes[1]
                x = x.reshape(B * T_frames, 3, shapes[-2], shapes[-1])
            elif 1 in c_positions:
                # [B, C, T, H, W] — NCTHW
                B, T_frames = shapes[0], shapes[2]
                x = x.permute(0, 2, 1, 3, 4).reshape(B * T_frames, 3, shapes[-2], shapes[-1])
            else:
                B, T_frames = shapes[0], shapes[2]
                x = x.reshape(B * T_frames, 3, shapes[-2], shapes[-1])
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        # Ensure input is 224×224 for ViT backbone
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224),
                                           mode='bilinear', align_corners=False)

        return x, B, T_frames

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Uses ViT's built-in patch_embed + blocks, with adapters inserted
        after each block. The adapter processes the intermediate token
        representation in a spatio-temporal manner.
        """
        x, B, T = self._parse_input(x)
        BT, C, H, W = x.shape
        spatial_H = H // self.backbone.patch_embed.patch_size[0]
        spatial_W = W // self.backbone.patch_embed.patch_size[1]

        # Patch embedding
        x = self.backbone.patch_embed(x)  # [BT, N, D]
        N = x.shape[1]
        D = x.shape[2]

        # CLS token + pos embed
        cls_token = self.backbone.cls_token.expand(BT, -1, -1)  # [BT, 1, D]
        x = torch.cat([cls_token, x], dim=1)  # [BT, N+1, D]
        x = x + self.backbone.pos_embed
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x)
        if hasattr(self.backbone, 'norm_pre'):
            x = self.backbone.norm_pre(x)

        # Apply blocks with adapters
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            # Insert adapter after each block
            x = self.adapters[i](x, spatial_H, spatial_W, T)

        if hasattr(self.backbone, 'norm'):
            x = self.backbone.norm(x)

        # Mean pool across all tokens
        feats = x.mean(dim=1)  # [BT, D]

        # Temporal pool: [BT, D] → [B, D]
        feats = feats.view(B, T, D).mean(dim=1)

        feats = self.cls_norm(feats)
        feats = self.dropout(feats)
        logits = self.fc(feats)

        if B == 1 and logits.dim() == 2:
            logits = logits.squeeze(0)

        return {"logits": logits}