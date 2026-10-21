"""MViTv2 — multiscale vision transformer for action recognition.

Uses timm's ``mvitv2_small/base/large`` as a per-frame 2D backbone,
followed by temporal average pooling and FC classifier.

Also supports Hiera (hiera_tiny/small/base) — a newer sparse
hierarchical ViT that serves as an improved MViTv2 successor.
"""

from __future__ import annotations

from elysia_cv.models.action_recognition.tsn.model import TSN


class MViT(TSN):
    """MViTv2 / Hiera backbone + temporal pooling + FC classifier.

    Thin wrapper around :class:`TSN` — reuses the same forward logic.
    Defaults to ``mvitv2_small`` (34M params, good speed/accuracy trade-off).

    Also supports Hiera models: ``hiera_tiny_224`` (27M), ``hiera_small_224`` (34M),
    ``hiera_base_224`` (51M).
    """

    def __init__(
        self,
        backbone_name: str = "mvitv2_small",
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )