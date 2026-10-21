"""TSN (Temporal Segment Network) — lightweight action recognition.

Architecture::

    [T, C, H, W] ──► 2D Backbone (per-frame) ──► [T, D]
                  ──► Temporal Average Pool ──► [D]
                  ──► FC Classifier ──► [num_classes]

Supports any timm backend (resnet50, efficientnet_b0, vit_*, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import timm


class TSN(nn.Module):
    """Temporal Segment Network.

    Args:
        backbone_name: timm model name (e.g. ``"resnet50"``).
        num_classes: Number of action classes.
        pretrained: Load ImageNet-pretrained weights.
        dropout: Dropout rate before the classifier head.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Build backbone, strip classifier
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg",
        )

        # Infer feature dimension
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat = self.backbone(dummy)
            self._feat_dim = int(feat.shape[-1])

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self._feat_dim, num_classes)

        self._loss_fn: nn.Module | None = None

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[T, C, H, W]`` (NCHW format).

        Returns:
            Dict with key ``"logits"`` — shape ``[num_classes]`` for single
            clip, or ``[B, num_classes]`` if batched.
        """
        # x: [T, C, H, W] (single clip) or [B, T, C, H, W] (batched)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            merge_batch = True
        elif x.dim() == 4:
            # Single clip: treat as batch=1
            B, T = 1, x.shape[0]
            x = x.view(B * T, x.shape[1], x.shape[2], x.shape[3])
            merge_batch = True
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        feats = self.backbone(x)            # [B*T, D]
        feats = self.dropout(feats)
        feats = feats.view(B, T, -1)        # [B, T, D]

        # Temporal average pool
        feats = feats.mean(dim=1)           # [B, D]

        # Squeeze batch dim for single-clip input
        if B == 1 and feats.shape[0] == 1:
            feats = feats.squeeze(0)        # [D]

        logits = self.fc(feats)             # [B, num_classes] or [num_classes]
        return {"logits": logits}