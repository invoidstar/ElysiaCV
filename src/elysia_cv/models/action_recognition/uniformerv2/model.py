"""UniFormerV2: CNN early stages + ViT later stages for video.

Reference: "UniFormerV2: Spatiotemporal Learning by Arming Image ViTs
with Video UniFormer" (CVPR 2023, Li et al.)

Simplified implementation:
  - Stage 1-2: 3D CNN blocks for efficient spatio-temporal modeling
  - Stage 3-4: Standard ViT blocks (from timm) for high-level reasoning
  - Temporal downsampling between CNN stages to reduce compute

Input:  [B, C, T, H, W]  (NCDHW format)
Output: [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class _Conv3DBlock(nn.Module):
    """3D CNN block: Conv3D → BN → ReLU."""

    def __init__(self, in_c: int, out_c: int, k: int = 3,
                 s: int | tuple = 1, p: int | tuple = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UniformerV2(nn.Module):
    """UniFormerV2 video classification.

    Args:
        num_classes:  Number of action classes.
        image_size:   Input spatial size (default 224).
        dropout:      Dropout rate for classification head.
        cnn_channels: Base channels for CNN stages (doubled each stage).
        vit_backbone: timm ViT backbone used for stage 3-4.
    """

    def __init__(
        self,
        num_classes: int = 101,
        image_size: int = 224,
        dropout: float = 0.5,
        cnn_channels: int = 64,
        vit_backbone: str = "vit_small_patch16_224",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ── Stage 1: 3D CNN ──
        # [B, 3, T, H, W] → [B, c, T/2, H/4, W/4]
        c = cnn_channels
        self.stem = nn.Sequential(
            _Conv3DBlock(3, c, k=(3, 7, 7), s=(1, 2, 2), p=(1, 3, 3)),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # Stage 2: CNN blocks
        self.stage2 = nn.Sequential(
            _Conv3DBlock(c, c * 2, k=3, s=1, p=1),
            _Conv3DBlock(c * 2, c * 2, k=3, s=1, p=1),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),  # temporal+spatial down
        )

        # ── ViT backbone for stage 3-4 ──
        # Flatten [B, c*2, T', H', W'] → [B, T', c*2, H', W'] for per-frame ViT
        self.vit_feat_dim = c * 2
        self.vit = timm.create_model(vit_backbone, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim

        # Project CNN output to ViT embed_dim
        self.proj = nn.Conv3d(c * 2, 3, kernel_size=1)  # cnn→3 for ViT RGB input

        # ── Classification head ──
        self.cls_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.embed_dim, num_classes)

        self.feat_dim = self.embed_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass. Input: ``[B, C, T, H, W]`` (NCDHW)."""
        if x.dim() == 4:
            # [T, C, H, W] → [1, C, T, H, W]
            x = x.unsqueeze(0).permute(0, 2, 1, 3, 4)
            single_batch = True
        elif x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4)
            single_batch = False
        else:
            single_batch = False

        B, C, T, H, W = x.shape

        # Stage 1
        x = self.stem(x)              # [B, c, T, H/4, W/4]

        # Stage 2
        x = self.stage2(x)            # [B, 2c, T/2, H/8, W/8]

        # Project to RGB for ViT processing
        _, c2, t2, h2, w2 = x.shape
        x_rgb = self.proj(x)          # [B, 3, T', H', W']

        # Per-frame ViT: [B, 3, T, H, W] → [B*T, 3, H, W]
        x_rgb = x_rgb.permute(0, 2, 1, 3, 4).reshape(B * t2, 3, h2, w2)

        # Interpolate to 224
        if h2 != 224:
            x_rgb = nn.functional.interpolate(x_rgb, size=(224, 224),
                                               mode='bilinear', align_corners=False)

        feats = self.vit(x_rgb)      # [B*T, D]
        feats = feats.view(B, t2, -1).mean(dim=1)  # [B, D]

        feats = self.cls_norm(feats)
        feats = self.dropout(feats)
        logits = self.fc(feats)

        if single_batch:
            logits = logits.squeeze(0)

        return {"logits": logits}