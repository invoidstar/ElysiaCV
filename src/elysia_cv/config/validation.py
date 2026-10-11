"""Cross-section config consistency validation.

Validates relationships BETWEEN sections that can't be checked at the
section-dataclass level.  Separated from config.py to keep the core
dataclass lean.
"""

from __future__ import annotations

from elysia_cv.config.config import CVConfig


def validate_cross_section(config: CVConfig) -> list[str]:
    """Validate cross-section consistency; return a list of error messages.

    An empty list means the config is valid.  This is intentionally a
    separate pass so it can be called from preflight / dry-run tools
    without importing heavy model / dataset code.
    """
    errors: list[str] = []

    # train stage: schedule steps should be large enough
    if config.run.stage == "train":
        if config.schedule.steps < 100:
            errors.append(
                f"schedule.steps={config.schedule.steps} seems too small for training"
            )
        if config.schedule.warmup_steps >= config.schedule.steps:
            errors.append(
                f"schedule.warmup_steps ({config.schedule.warmup_steps}) >= "
                f"schedule.steps ({config.schedule.steps})"
            )

    # test stage requires eval metrics
    if config.run.stage == "test":
        if not config.eval.metrics:
            errors.append("eval.metrics is required for test stage")
        if config.eval.eval_freq > 0:
            # eval_freq is a train-time param; harmless override
            pass

    # model pretrained_path existence (best-effort: path may be remote/hf)
    if config.model.pretrained_path:
        from pathlib import Path
        p = Path(config.model.pretrained_path)
        if p.exists() and not p.is_file():
            errors.append(f"model.pretrained_path is not a file: {p}")

    return errors


def validate_or_raise(config: CVConfig) -> None:
    """Like :func:`validate_cross_section`, but raises on first error."""
    errs = validate_cross_section(config)
    if errs:
        raise ValueError("config validation failed:\n  " + "\n  ".join(errs))