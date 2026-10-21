"""TSM (Temporal Shift Module) — action recognition with zero extra parameters.

Architecture::
    [T, C, H, W] ──► 2D Backbone (no global pool) ──► [T, C', H', W']
                  ──► Temporal Shift (shift channels along T dim) ──►
                  ──► Global Avg Pool ──► [T, D]
                  ──► Temporal Mean ──► [D]
                  ──► FC Classifier ──► [num_classes]

Key difference from TSN: temporal_shift() applied to intermediate feature
maps BEFORE pooling, enabling inter-frame information exchange at zero
additional parameter cost.  Typically +3-5% over TSN on UCF-101.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import timm


class TSM(nn.Module):
    """Temporal Shift Module.

    Args:
        backbone_name: timm model name (e.g. ``"resnet50"``).
        num_classes: Number of action classes.
        pretrained: Load ImageNet-pretrained weights.
        dropout: Dropout rate before the classifier head.
        shift_div: Fraction of channels to shift (default 8 → 1/8 left + 1/8 right).
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.5,
        shift_div: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.shift_div = shift_div
        self.dropout_rate = dropout

        # Backbone WITHOUT global pooling — we need spatial feature maps
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="",
        )

        # Infer feature dimension from a dummy forward
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat_map = self.backbone(dummy)  # [1, C', H', W']
            self._feat_dim = int(feat_map.shape[1])

        # Post-processing
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self._feat_dim, num_classes)

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def _temporal_shift(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """Shift part of channels along temporal dimension.

        Args:
            x: [B*T, C, H, W] feature maps
            T: number of frames

        Returns:
            [B*T, C, H, W] with channels shifted
        """
        nt, c, h, w = x.shape
        B = nt // T
        x = x.view(B, T, c, h, w)

        fold = max(1, c // self.shift_div)
        out = torch.zeros_like(x)
        # First 1/shift_div channels: shift left (forward in time)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # Next 1/shift_div channels: shift right (backward in time)
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        # Remaining channels: keep unchanged
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]

        return out.view(nt, c, h, w)

    def forward(self, x: torch.Tensor, shift: bool = True) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor ``[T, C, H, W]`` or ``[B, T, C, H, W]``.
            shift: Apply temporal shift (True for training, False for pure TSN mode).

        Returns:
            Dict with ``"logits"`` key.
        """
        if x.dim() == 5:
            B, T = x.shape[0], x.shape[1]
            x = x.view(B * T, *x.shape[2:])
        elif x.dim() == 4:
            B, T = 1, x.shape[0]
            x = x.view(B * T, *x.shape[1:])
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

        feats = self.backbone(x)                # [B*T, C', H', W']

        if shift and T > 1:
            feats = self._temporal_shift(feats, T)  # [B*T, C', H', W']

        feats = self.avgpool(feats)             # [B*T, C', 1, 1]
        feats = feats.view(B, T, -1)            # [B, T, D]
        feats = feats.mean(dim=1)               # [B, D]
        feats = self.dropout(feats)             # [B, D]

        if B == 1 and feats.shape[0] == 1:
            feats = feats.squeeze(0)            # [D]

        logits = self.fc(feats)                 # [B, num_classes] or [num_classes]
        return {"logits": logits}