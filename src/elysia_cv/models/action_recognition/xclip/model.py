"""X-CLIP: CLIP-pretrained ViT + temporal cross-attention for video.

Uses timm's CLIP-pretrained ViT as a frozen per-frame feature extractor, then
applies lightweight temporal attention across frames before classification.

Reference: "Expanding Language-Image Pretrained Models for General Video
Recognition" (ECCV 2022, Ni et al.)

Simplified implementation:
  1. CLIP ViT → per-frame [CLS] tokens → [B, T, D]
  2. Temporal Transformer (1-2 layers of self-attention across T)
  3. Mean-pool [CLS] tokens → FC classification head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class XCLIP(nn.Module):
    """X-CLIP video classification model.

    Args:
        backbone_name:  timm CLIP ViT backbone name, e.g. ``"vit_base_patch16_clip_224"``.
        num_classes:    Number of action classes.
        pretrained:     Load CLIP-pretrained weights for the backbone.
        num_temp_layers: Number of temporal transformer layers (default 2).
        dropout:        Dropout rate for the classification head.
        freeze_backbone: Whether to freeze the CLIP backbone (default True).
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_clip_224",
        num_classes: int = 101,
        pretrained: bool = True,
        num_temp_layers: int = 2,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # ── CLIP ViT backbone ──
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,           # strip classification head
        )
        embed_dim = self.backbone.embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Temporal transformer ──
        self.temporal_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=min(8, embed_dim // 64),
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_temp_layers
        )

        # ── Classification head ──
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.feat_dim = embed_dim
        self._loss_fn: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``[B, T, C, H, W]``, ``[T, C, H, W]``, or ``[B, C, T, H, W]`` tensor.

        Returns:
            Dict with ``"logits"`` key, shape ``[B, num_classes]``.
        """
        # Robust dim detection: find where C==3 channel dim is
        if x.dim() == 4:
            # [T, C, H, W] → treat T as batch
            T, C, H, W = x.shape
            if C == 3:
                x = x.view(T, C, H, W)
            elif T == 3:
                x = x.permute(1, 0, 2, 3)  # [C,T,H,W] → [T,C,H,W]
            else:
                x = x  # hope for the best
            single_batch = True
            B = 1
            T_frames = x.shape[0]
        elif x.dim() == 5:
            shapes = list(x.shape)
            # Find C=3 position
            c_positions = [i for i, s in enumerate(shapes) if s == 3]
            if 2 in c_positions:
                # [B, T, C, H, W] → NCHW stacked by DataLoader
                B, T_frames, C, H, W = shapes
                x = x.reshape(B * T_frames, C, H, W)
            elif 1 in c_positions:
                # [B, C, T, H, W] → NCTHW
                B, C, T_frames, H, W = shapes
                x = x.permute(0, 2, 1, 3, 4).reshape(B * T_frames, C, H, W)
            else:
                x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
                T_frames = x.shape[0]
            single_batch = False
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        # Per-frame CLIP features → [B*T, D]
        with torch.set_grad_enabled(not all(p.requires_grad is False for p in self.backbone.parameters())):
            if hasattr(self.backbone, 'forward_features'):
                feats = self.backbone.forward_features(x)
                # For ViT, forward_features returns [N, P+1, D] (patches + cls)
                if feats.dim() == 3 and feats.shape[1] > 1:
                    # Take CLS token + global pool for robustness
                    feats = feats[:, 0, :] + feats[:, 1:, :].mean(dim=1)
                else:
                    feats = feats.view(feats.shape[0], -1)
            else:
                feats = self.backbone(x)

        # Reshape: [B*T, D] → [B, T_frames, D]
        if single_batch:
            feats = feats.unsqueeze(0)  # [1, T_frames, D]
        else:
            feats = feats.view(B, T_frames, -1)

        # Temporal attention
        feats = self.temporal_norm(feats)
        feats = self.temporal_transformer(feats)  # [B, T_frames, D]

        # Pool across frames
        feats = feats.mean(dim=1)  # [B, D]

        # Classification
        logits = self.fc(feats)

        if single_batch:
            logits = logits.squeeze(0)

        return {"logits": logits}