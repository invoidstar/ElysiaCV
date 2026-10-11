"""Loss functions — auto-registered via LOSSES registry."""

from __future__ import annotations

from elysia_cv.losses.cross_entropy import CrossEntropyLoss, FocalLoss  # noqa: F401
from elysia_cv.losses.smooth_l1 import SmoothL1Loss, IoULoss, GIoULoss  # noqa: F401
from elysia_cv.losses.dice_loss import DiceLoss, CombinedSegLoss  # noqa: F401