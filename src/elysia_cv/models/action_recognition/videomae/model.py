"""VideoMAE wrapper — huggingface transformers integration.

Reference: "VideoMAE: Masked Autoencoders are Data-Efficient Learners
for Self-Supervised Video Pre-Training" (NeurIPS 2022, Tong et al.)

Uses ``transformers.VideoMAEForVideoClassification`` with pretrained weights.

Input:  [B, C, T, H, W] or [B, T, C, H, W]
Output: [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VideoMAE(nn.Module):
    """VideoMAE video classification model.

    Args:
        model_name:  HuggingFace model ID or path.
                     Default: ``"MCG-NJU/videomae-base"``.
        num_classes: Number of action classes.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        num_classes: int = 101,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        from transformers import VideoMAEForVideoClassification

        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``[B, C, T, H, W]`` (NCDHW) or ``[B, T, C, H, W]``.

        Returns:
            Dict with ``"logits"`` key.
        """
        if x.dim() == 4:
            x = x.permute(1, 0, 2, 3).unsqueeze(0)  # [T,C,H,W] → [1,T,C,H,W]
        elif x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] → [B,T,C,H,W]

        # Resize to 224×224 if needed
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(
                x.permute(0, 2, 1, 3, 4),  # [B,T,C,H,W] → [B*T,C,H,W]
                size=(224, 224), mode='bilinear', align_corners=False
            ).view(x.shape[0], x.shape[1], x.shape[2], 224, 224).permute(0, 2, 1, 3, 4)

        out = self.model(x)
        return {"logits": out.logits}