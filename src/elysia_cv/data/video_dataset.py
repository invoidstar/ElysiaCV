"""VideoDataset — Map-style PyTorch dataset for action recognition.

Supports five dataset formats:

* ``"kinetics"``     — ``root/{split}/{class_name}/{video}.mp4`` folder layout.
* ``"ucf101"``       — ``root/{class_name}/{video}.avi`` + split file.
* ``"json"``         — ``root/{split}_annotations.json`` with video→label mapping.
* ``"csv"``          — ``root/{split}_annotations.csv`` (Kinetics CSV format).
* ``"kinetics_csv"`` — alias for "csv" with Kinetics column names.

Usage::

    dataset = VideoDataset(
        root="/data/kinetics400",
        split="train",
        format="kinetics",
        transforms=[...],
    )
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from elysia_cv.data.dataset import BaseDataset
from elysia_cv.data.transforms import build_pipeline


class VideoDataset(BaseDataset):
    """Map-style video dataset for action recognition.

    Args:
        root:        Root directory of the dataset.
        split:       ``"train"``, ``"val"``, or ``"test"``.
        format:      Dataset format (see module docstring).
        csv_file:    Path to CSV annotation file (for ``format="csv"``).
                     Defaults to ``root/{split}_annotations.csv``.
        transforms:  List of transform config dicts (passed to ``build_pipeline``).
    """

    def __init__(
        self,
        root: str = "",
        split: str = "train",
        format: str = "kinetics",
        csv_file: str | None = None,
        transforms: list[dict[str, Any]] | None = None,
    ) -> None:
        self.format = format
        self._csv_file = csv_file
        self._classes: list[str] = []
        self._class_to_idx: dict[str, int] = {}
        super().__init__(root=root, split=split, transforms=transforms)

    # ------------------------------------------------------------------
    # Annotation loading — routed by format
    # ------------------------------------------------------------------

    def _load_annotations(self) -> list[dict[str, Any]]:
        if self.format in ("kinetics",):
            return self._load_kinetics()
        elif self.format == "ucf101":
            return self._load_ucf101()
        elif self.format == "json":
            return self._load_json()
        elif self.format in ("csv", "kinetics_csv"):
            return self._load_csv()
        else:
            raise ValueError(
                f"Unknown VideoDataset format: {self.format!r}. "
                f"Expected 'kinetics', 'ucf101', 'json', or 'csv'."
            )

    def _load_kinetics(self) -> list[dict[str, Any]]:
        """Kinetics format: ``root/{split}/{class_name}/{video}.mp4``."""
        split_dir = self.root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Kinetics {self.split} dir not found: {split_dir}")

        entries = sorted(
            d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        self._classes = [d.name for d in entries]
        self._class_to_idx = {name: i for i, name in enumerate(self._classes)}

        samples: list[dict[str, Any]] = []
        for class_dir in entries:
            class_idx = self._class_to_idx[class_dir.name]
            for ext in ("*.mp4", "*.avi", "*.mkv", "*.webm", "*.mov"):
                for video_path in sorted(class_dir.glob(ext)):
                    samples.append({
                        "video_path": str(video_path),
                        "gt_label": class_idx,
                    })
        return samples

    def _load_ucf101(self) -> list[dict[str, Any]]:
        """UCF-101 format: ``root/{class}/{video}.avi`` + split file.

        Split files are expected at:
            ``root/ucfTrainTestlist/{split}list01.txt``

        Each line: ``{class_name}/{video_filename}  {label_id}``
        where label_id is 1-indexed.
        """
        split_map = {"train": "train", "val": "test", "test": "test"}
        split_suffix = split_map.get(self.split, self.split)
        split_file = self.root / "ucfTrainTestlist" / f"{split_suffix}list01.txt"

        if not split_file.is_file():
            raise FileNotFoundError(
                f"UCF-101 split file not found: {split_file}"
            )

        # First pass: collect all unique class names
        class_set: set[str] = set()
        entries: list[tuple[str, int]] = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                rel_path, label_str = parts
                # Rel path is like "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
                class_name = rel_path.split("/")[0]
                label_id = int(label_str) - 1  # convert to 0-indexed
                class_set.add(rel_path.split("/")[0])
                entries.append((rel_path, label_id))

        self._classes = sorted(class_set)
        self._class_to_idx = {name: i for i, name in enumerate(self._classes)}

        # Build label remap: split-file label → canonical class idx
        # UCF-101 split files use 1-based alphabetical ordering
        label_map: dict[int, int] = {}
        for rel_path, label_id in entries:
            class_name = rel_path.split("/")[0]
            label_map[label_id] = self._class_to_idx[class_name]

        # Second pass: build samples
        samples: list[dict[str, Any]] = []
        for rel_path, label_id in entries:
            video_path = self.root / rel_path
            if video_path.is_file():
                samples.append({
                    "video_path": str(video_path),
                    "gt_label": label_map[label_id],
                })
            else:
                # Try alternate extensions
                stem = video_path.stem
                parent = video_path.parent
                found = False
                for ext in (".mp4", ".avi", ".mkv", ".webm", ".mov"):
                    alt = parent / (stem + ext)
                    if alt.is_file():
                        samples.append({
                            "video_path": str(alt),
                            "gt_label": label_map[label_id],
                        })
                        found = True
                        break
                if not found:
                    print(f"  [VideoDataset] WARNING: video not found: {video_path}")

        return samples

    def _load_json(self) -> list[dict[str, Any]]:
        """Generic JSON format: ``root/{split}_annotations.json``.

        Expected format:
            {
                "video1.mp4": {"label": "class_name"},
                "video2.mp4": {"label": "class_name"},
            }
        """
        annot_file = self.root / f"{self.split}_annotations.json"
        if not annot_file.is_file():
            raise FileNotFoundError(f"Annotation file not found: {annot_file}")

        with open(annot_file, "r") as f:
            data = json.load(f)

        # Collect classes
        class_set: set[str] = set()
        for info in data.values():
            class_set.add(str(info.get("label", info) if isinstance(info, dict) else info))

        self._classes = sorted(class_set)
        self._class_to_idx = {name: i for i, name in enumerate(self._classes)}

        samples: list[dict[str, Any]] = []
        for video_name, info in data.items():
            video_path = self.root / video_name
            if video_path.is_file():
                label = info.get("label", info) if isinstance(info, dict) else info
                samples.append({
                    "video_path": str(video_path),
                    "gt_label": self._class_to_idx[str(label)],
                })
        return samples

    def _load_csv(self) -> list[dict[str, Any]]:
        """Kinetics CSV format: CSV file with video metadata.

        Expected CSV columns: ``label``, ``youtube_id``, ``time_start``,
        ``time_end``, ``split`` (Kinetics-400 format).

        Videos are expected at ``root/{label}/{youtube_id}_{time_start:06d}_{time_end:06d}.mp4``
        (the standard Kinetics naming convention).

        If ``csv_file`` is provided, uses that path; otherwise defaults to
        ``root/{split}_annotations.csv``.
        """
        if self._csv_file:
            csv_path = Path(self._csv_file)
        else:
            csv_path = self.root / f"{self.split}_annotations.csv"

        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV annotation file not found: {csv_path}")

        # Parse CSV — handle both header variants
        label_to_idx: dict[str, int] = {}
        samples: list[dict[str, Any]] = []
        next_label_idx = 0

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")

            # Normalize column names (strip whitespace, lowercase)
            fieldnames = [fn.strip().lower() for fn in reader.fieldnames]

            for row in reader:
                # Normalize row keys
                row = {k.strip().lower(): v.strip() for k, v in row.items()}

                label = row.get("label", "")
                if not label:
                    continue

                # Skip rows not in this split (if split column exists)
                row_split = row.get("split", "")
                if row_split and row_split != self.split:
                    continue

                # Assign class index
                if label not in label_to_idx:
                    label_to_idx[label] = next_label_idx
                    next_label_idx += 1

                # Build video filename: standard Kinetics naming
                # Pattern: {label}/{youtube_id}_{time_start:06d}_{time_end:06d}.mp4
                youtube_id = row.get("youtube_id", "")
                if youtube_id:
                    try:
                        t_start = int(float(row.get("time_start", 0)))
                        t_end = int(float(row.get("time_end", 0)))
                        filename = f"{youtube_id}_{t_start:06d}_{t_end:06d}.mp4"
                    except (ValueError, TypeError):
                        filename = f"{youtube_id}.mp4"
                else:
                    # Fallback: use any video_id field
                    video_id = row.get("video_id", row.get("filename", ""))
                    if video_id:
                        filename = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
                    else:
                        continue

                video_path = self.root / label / filename
                if video_path.is_file():
                    samples.append({
                        "video_path": str(video_path),
                        "gt_label": label_to_idx[label],
                    })
                else:
                    # Try without label subdir (flat layout)
                    alt_path = self.root / filename
                    if alt_path.is_file():
                        samples.append({
                            "video_path": str(alt_path),
                            "gt_label": label_to_idx[label],
                        })
                    else:
                        # Try alternative extensions
                        stem = video_path.stem
                        parent = video_path.parent
                        found = False
                        for ext in (".mp4", ".avi", ".mkv", ".webm", ".mov"):
                            alt = parent / (stem + ext)
                            if alt.is_file():
                                samples.append({
                                    "video_path": str(alt),
                                    "gt_label": label_to_idx[label],
                                })
                                found = True
                                break
                        if not found and alt_path.parent.is_dir():
                            pass  # silently skip missing videos

        self._classes = sorted(label_to_idx.keys(), key=lambda x: label_to_idx[x])
        self._class_to_idx = label_to_idx
        return samples

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def classes(self) -> list[str]:
        return self._classes

    # ------------------------------------------------------------------
    # __getitem__ — keys expected by video transforms
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index].copy()
        for transform in self._transforms:
            sample = transform(sample)
        return sample