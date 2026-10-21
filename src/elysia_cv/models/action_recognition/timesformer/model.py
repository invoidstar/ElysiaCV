"""TimeSformer wrapper — huggingface transformers integration.

Reference: "Is Space-Time Attention All You Need for Video Understanding?"
(ICML 2021, Bertasius et al.)

Uses ``transformers.TimesformerModel`` / ``TimesformerForVideoClassification``
with pretrained Kinetics-400 weights.

Input:  [B, C, T, H, W] or [B, T, C, H, W]
Output: [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeSformer(nn.Module):
    """TimeSformer video classification model.

    Args:
        model_name:   HuggingFace model ID or path.
                       Default: ``"facebook/timesformer-base-finetuned-k400"``.
        num_classes:  Number of action classes.
        pretrained:   Load pretrained weights from HuggingFace hub.
    """

    def __init__(
        self,
        model_name: str = "facebook/timesformer-base-finetuned-k400",
        num_classes: int = 101,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        from transformers import TimesformerForVideoClassification

        self.model = TimesformerForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        # The model expects [B, T, C, H, W] (BTC format)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: ``[B, C, T, H, W]`` (NCDHW) or ``[B, T, C, H, W]`` (BTCHW).

        Returns:
            Dict with ``"logits"`` key.
        """
        # Convert to transformers expected format: [B, T, C, H, W]
        if x.dim() == 4:
            x = x.permute(1, 0, 2, 3).unsqueeze(0)  # [T,C,H,W] → [1,T,C,H,W]
        elif x.dim() == 5 and x.shape[1] == 3 and x.shape[2] != 3:
            x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] → [B,T,C,H,W]

        out = self.model(x)
        return {"logits": out.logits}