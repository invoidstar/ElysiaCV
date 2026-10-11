"""Detection heads — registered via HEADS registry."""

from __future__ import annotations

import torch
import torch.nn as nn


from elysia_cv.core.registry import HEADS


@HEADS.register_module()
class DetHead(nn.Module):
    """Simple anchor-free detection head (similar to FCOS head).

    Predicts classification scores and bounding box offsets per spatial
    location across all pyramid levels.

    Args:
        num_classes: Number of object classes (excluding background).
        in_channels: Input channels per pyramid level.
        feat_channels: Internal feature channels.
        stacked_convs: Number of conv layers per branch.
    """

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: int = 256,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Shared feature tower
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(
                nn.Conv2d(chn, feat_channels, kernel_size=3, padding=1)
            )
            self.cls_convs.append(nn.GroupNorm(32, feat_channels))
            self.cls_convs.append(nn.ReLU(inplace=True))
            self.reg_convs.append(
                nn.Conv2d(chn, feat_channels, kernel_size=3, padding=1)
            )
            self.reg_convs.append(nn.GroupNorm(32, feat_channels))
            self.reg_convs.append(nn.ReLU(inplace=True))

        # Prediction branches
        self.cls_out = nn.Conv2d(feat_channels, num_classes, kernel_size=3, padding=1)
        self.reg_out = nn.Conv2d(feat_channels, 4, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Bias init for classification
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_out.bias, bias_value)

    def forward(self, feats: list[torch.Tensor] | torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Forward across all FPN levels.

        Args:
            feats: List of feature maps from FPN (each [B, C, H, W]).

        Returns:
            dict with "cls_score" and "bbox_pred" (lists of tensors per level).
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        cls_scores: list[torch.Tensor] = []
        bbox_preds: list[torch.Tensor] = []

        for feat in feats:
            cls_feat = feat
            for layer in self.cls_convs:
                cls_feat = layer(cls_feat)
            cls_scores.append(self.cls_out(cls_feat))

            reg_feat = feat
            for layer in self.reg_convs:
                reg_feat = layer(reg_feat)
            bbox_preds.append(self.reg_out(reg_feat))

        return {"cls_score": cls_scores, "bbox_pred": bbox_preds}