"""Cassava Leaf Disease Dataset — reads train_split.csv / val_split.csv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CassavaDataset(Dataset):
    """Cassava Leaf Disease Classification dataset.

    Args:
        data_root: Root directory with ``train_images/`` and CSV files.
        split: ``train`` or ``val`` — reads ``{split}_split.csv``.
        transform: Optional image transform / augmentation pipeline.

    Returns:
        dict with ``inputs`` (PIL Image), ``gt_label`` (int), ``image_id`` (str).
    """

    _DISEASE_NAMES = [
        "Cassava Bacterial Blight (CBB)",
        "Cassava Brown Streak Disease (CBSD)",
        "Cassava Green Mottle (CGM)",
        "Cassava Mosaic Disease (CMD)",
        "Healthy",
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Any = None,
        pipeline: Any = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = Path(data_root)
        self.split = split
        # Support both legacy ``transform`` and ``pipeline`` (prefer pipeline)
        self.pipeline = pipeline if pipeline is not None else transform

        import pandas as pd

        csv_path = self.root / f"{split}_split.csv"
        self.df = pd.read_csv(csv_path)
        self.image_ids = self.df["image_id"].tolist()
        self.labels = self.df["label"].tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id = self.image_ids[idx]
        label = self.labels[idx]

        # Image
        img_path = self.root / "train_images" / img_id
        img = Image.open(img_path).convert("RGB")

        if self.pipeline is not None:
            img = self.pipeline(img)

        return {
            "inputs": img,
            "gt_label": label,
            "image_id": img_id,
        }

    @classmethod
    def num_classes(cls) -> int:
        return 5

    @classmethod
    def class_names(cls) -> list[str]:
        return list(cls._DISEASE_NAMES)