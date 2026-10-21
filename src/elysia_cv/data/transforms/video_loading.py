"""Video loading transforms for action recognition.

These transforms operate on the ``results`` dict protocol, co-existing
with existing spatial transforms.  Video-specific keys used:

    ``video_path``    – path to the video file (set by dataset)
    ``vr``            – decord.VideoReader handle (set by DecordInit)
    ``total_frames``  – total number of frames in the video
    ``frame_inds``    – selected frame indices (set by SampleFrames)
    ``imgs``          – decoded frames as (T, H, W, C) numpy array

After DecordDecode, spatial transforms operate on ``imgs`` and
adjust ``img_shape`` accordingly.  FormatShape converts to the
layout expected by the model.
"""

from __future__ import annotations

import numpy as np
import os

from elysia_cv.core.registry import TRANSFORMS


# ── DecordInit ──────────────────────────────────────────────────────────────

@TRANSFORMS.register_module()
class DecordInit:
    """Open a video file with decord and store the VideoReader handle.

    Expects ``results["video_path"]``.

    Args:
        num_threads: Number of CPU threads for decoding (default 2).
    """

    def __init__(self, num_threads: int = 2) -> None:
        self.num_threads = num_threads
        os.environ.setdefault("DECORD_EOF_RETRY_MAX", "4096")

    def __call__(self, results: dict) -> dict:
        from decord import VideoReader, cpu
        vr = VideoReader(results["video_path"], ctx=cpu(self.num_threads))
        results["vr"] = vr
        results["total_frames"] = len(vr)
        results["avg_fps"] = vr.get_avg_fps()
        return results


# ── SampleFrames ────────────────────────────────────────────────────────────

@TRANSFORMS.register_module()
class SampleFrames:
    """Select frame indices from a video.

    Args:
        num_frames:   Number of frames to sample (e.g. 8, 16, 32).
        sampling:     Strategy:
                      ``"random"``  — random starting offset + uniform spacing (train)
                      ``"uniform"`` — strict uniform spacing (val/test)
                      ``"dense"``   — contiguous segment (for SlowFast / 3D CNN)
        clip_duration: If set (seconds), only sample from a random
                      segment of this length (useful for long videos).
    """

    def __init__(self, num_frames: int = 8, sampling: str = "random",
                 clip_duration: float | None = None) -> None:
        self.num_frames = num_frames
        self.sampling = sampling
        self.clip_duration = clip_duration

    def __call__(self, results: dict) -> dict:
        total = results["total_frames"]
        fps = results.get("avg_fps", 30.0)

        if self.clip_duration is not None:
            clip_len = int(self.clip_duration * fps)
            clip_len = min(clip_len, total)
            if self.sampling == "random":
                start = np.random.randint(0, total - clip_len + 1) if total > clip_len else 0
            else:
                start = 0
            total = clip_len
        else:
            start = 0

        if self.sampling == "random" and total >= self.num_frames:
            avg_interval = total / self.num_frames
            offsets = np.random.uniform(0, avg_interval, size=self.num_frames)
            base = np.arange(self.num_frames) * avg_interval
            indices = np.floor(base + offsets).astype(np.int64)
            indices = np.clip(indices, 0, total - 1)
        elif self.sampling == "dense" and total >= self.num_frames:
            max_start = total - self.num_frames
            s = np.random.randint(0, max_start + 1) if self.sampling != "uniform" else 0
            indices = np.arange(s, s + self.num_frames)
        else:  # uniform
            if total >= self.num_frames:
                indices = np.linspace(0, total - 1, self.num_frames, dtype=np.int64)
            else:
                # Pad: repeat the last frame
                indices = np.zeros(self.num_frames, dtype=np.int64)
                indices[:total] = np.arange(total)
                indices[total:] = total - 1

        if self.clip_duration is not None:
            indices += start

        results["frame_inds"] = indices
        results["num_clips"] = 1
        return results


# ── DecordDecode ────────────────────────────────────────────────────────────

@TRANSFORMS.register_module()
class DecordDecode:
    """Decode selected frame indices into a numpy array.

    Expects ``results["vr"]`` (decord VideoReader) and ``results["frame_inds"]``.
    Sets ``results["imgs"]`` as a (T, H, W, C) float32 array in RGB, range [0, 255].
    Also sets ``results["img_shape"]`` to (H, W) for compatibility with spatial transforms.
    """

    def __init__(self, backend: str = "decord") -> None:
        self.backend = backend

    def __call__(self, results: dict) -> dict:
        vr = results.pop("vr", None)
        frame_inds = results["frame_inds"]

        # Clamp indices to valid range
        frame_inds = np.clip(frame_inds, 0, len(vr) - 1)

        # decord returns [T, H, W, C] RGB uint8
        imgs = vr.get_batch(frame_inds).asnumpy()
        imgs = imgs.astype(np.float32)

        results["imgs"] = imgs
        results["img_shape"] = imgs.shape[1:3]  # (H, W)
        results["ori_shape"] = imgs.shape[1:3]
        results.pop("frame_inds", None)
        results.pop("total_frames", None)
        return results


# ── VideoSpatial transforms ─────────────────────────────────────────────────
# These are video-aware versions that operate on ``results["imgs"]``
# (a [T, H, W, C] array) by applying the same operation to every frame.


@TRANSFORMS.register_module()
class VideoResize:
    """Resize all frames to a target size.

    Args:
        scale: (width, height) tuple, or single integer for short-side scaling.
        keep_ratio: If True, resize short side to scale and pad to square.
    """

    def __init__(self, scale: int | tuple[int, int] = 224, keep_ratio: bool = False) -> None:
        self.scale = scale
        self.keep_ratio = keep_ratio

    def __call__(self, results: dict) -> dict:
        import cv2
        imgs = results["imgs"]  # [T, H, W, C]
        T, H, W, C = imgs.shape

        if isinstance(self.scale, (list, tuple)):
            target_w, target_h = int(self.scale[0]), int(self.scale[1])
        else:
            target_w = target_h = int(self.scale)

        if self.keep_ratio:
            # Resize short side, pad to square
            short = min(H, W)
            sc = target_h / short
            new_h, new_w = int(H * sc), int(W * sc)
            new_imgs = np.zeros((T, target_h, target_w, C), dtype=imgs.dtype)
            for t in range(T):
                resized = cv2.resize(imgs[t], (new_w, new_h))
                dh = (target_h - new_h) // 2
                dw = (target_w - new_w) // 2
                new_imgs[t, dh:dh + new_h, dw:dw + new_w] = resized
            imgs = new_imgs
        else:
            new_imgs = np.zeros((T, target_h, target_w, C), dtype=imgs.dtype)
            for t in range(T):
                new_imgs[t] = cv2.resize(imgs[t], (target_w, target_h))
            imgs = new_imgs

        results["imgs"] = imgs
        results["img_shape"] = imgs.shape[1:3]
        return results


@TRANSFORMS.register_module()
class VideoRandomCrop:
    """Random crop all frames to the same spatial location.

    Args:
        size: (height, width) tuple.
    """

    def __init__(self, size: int | tuple[int, int] = 224) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, results: dict) -> dict:
        imgs = results["imgs"]  # [T, H, W, C]
        _, H, W, _ = imgs.shape
        th, tw = self.size
        y = np.random.randint(0, max(1, H - th + 1))
        x = np.random.randint(0, max(1, W - tw + 1))
        imgs = imgs[:, y:y + th, x:x + tw, :]
        results["imgs"] = imgs
        results["img_shape"] = imgs.shape[1:3]
        return results


@TRANSFORMS.register_module()
class VideoCenterCrop:
    """Center crop all frames.

    Args:
        size: (height, width) tuple.
    """

    def __init__(self, size: int | tuple[int, int] = 224) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, results: dict) -> dict:
        imgs = results["imgs"]  # [T, H, W, C]
        _, H, W, _ = imgs.shape
        th, tw = self.size
        y = (H - th) // 2
        x = (W - tw) // 2
        imgs = imgs[:, y:y + th, x:x + tw, :]
        results["imgs"] = imgs
        results["img_shape"] = imgs.shape[1:3]
        return results


@TRANSFORMS.register_module()
class VideoRandomHorizontalFlip:
    """Randomly flip all frames horizontally with prob=0.5."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, results: dict) -> dict:
        if np.random.random() < self.prob:
            results["imgs"] = results["imgs"][:, :, ::-1, :]
        return results


@TRANSFORMS.register_module()
class VideoNormalize:
    """Normalize all frames to [0, 1] or using mean/std.

    Args:
        to_rgb: If True, convert float [0,255] to [0,1].
        mean: Channel-wise mean for (mean - img) / std normalization.
              If None, only scales to [0,1].
        std: Channel-wise std.
    """

    def __init__(self, mean: list[float] | None = None,
                 std: list[float] | None = None) -> None:
        self.mean = np.array(mean, dtype=np.float32) if mean else None
        self.std = np.array(std, dtype=np.float32) if std else None

    def __call__(self, results: dict) -> dict:
        imgs = results["imgs"].astype(np.float32)  # [T, H, W, C]
        imgs /= 255.0
        if self.mean is not None:
            imgs = (imgs - self.mean.reshape(1, 1, 1, -1))
        if self.std is not None:
            imgs = imgs / self.std.reshape(1, 1, 1, -1)
        results["imgs"] = imgs
        return results


# ── FormatShape ─────────────────────────────────────────────────────────────

@TRANSFORMS.register_module()
class FormatShape:
    """Convert ``imgs`` from [T, H, W, C] to the target memory layout.

    Args:
        input_format: ``"NCTHW"`` → [1, C, T, H, W]  (3D model)
                      ``"NCHW"``  → [T, C, H, W]    (2D model, T as batch)
    """

    def __init__(self, input_format: str = "NCTHW") -> None:
        self.input_format = input_format

    def __call__(self, results: dict) -> dict:
        imgs = results["imgs"]  # [T, H, W, C] float32
        if self.input_format == "NCTHW":
            # [T, H, W, C] → [1, C, T, H, W]
            imgs = imgs.transpose(3, 0, 1, 2)  # [C, T, H, W]
            imgs = imgs[np.newaxis, ...]        # [1, C, T, H, W]
        elif self.input_format == "NCHW":
            # [T, H, W, C] → [T, C, H, W]
            imgs = imgs.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"Unknown input_format: {self.input_format}")
        results["imgs"] = imgs
        return results


# ── PackVideoInputs ─────────────────────────────────────────────────────────

@TRANSFORMS.register_module()
class PackVideoInputs:
    """Pack video data into a batch-friendly ``inputs`` key.

    Sets ``results["inputs"]`` as a tensor from ``results["imgs"]``.
    Keeps ``gt_label`` in the results dict.
    """

    def __call__(self, results: dict) -> dict:
        import torch
        imgs = results.pop("imgs")
        results["inputs"] = torch.from_numpy(imgs.copy()).float()
        # Ensure gt_label is a scalar tensor
        if "gt_label" in results and not isinstance(results["gt_label"], torch.Tensor):
            results["gt_label"] = torch.tensor(results["gt_label"], dtype=torch.long)
        return results