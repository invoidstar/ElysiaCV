"""Top-level CVConfig frozen dataclass — the single source of truth.

Adapted from ElysiaRobot's ``NativeTrainingConfig`` pattern.
Sections are typed via ``sections.py``; loading and override merging
happen in ``loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from elysia_cv.config.sections import (
    DataSection,
    EngineSection,
    EvalSection,
    ModelSection,
    OptimSection,
    RunSection,
    ScheduleSection,
    TrackingSection,
)

_ALLOWED_STAGES = ("train", "test", "infer")
_ALLOWED_BACKENDS = ("native", "accelerate")
_ALLOWED_STRATEGIES = ("ddp", "fsdp", "single")


@dataclass(frozen=True)
class CVConfig:
    """Typed, validated configuration for one CV training / evaluation run."""

    run: RunSection = field(default_factory=RunSection)
    model: ModelSection = field(default_factory=ModelSection)
    data: DataSection = field(default_factory=DataSection)
    engine: EngineSection = field(default_factory=EngineSection)
    optim: OptimSection = field(default_factory=OptimSection)
    schedule: ScheduleSection = field(default_factory=ScheduleSection)
    tracking: TrackingSection = field(default_factory=TrackingSection)
    eval: EvalSection = field(default_factory=EvalSection)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CVConfig":
        def _section(key: str, builder, default):
            value = mapping.get(key)
            return builder(value) if isinstance(value, Mapping) else default

        return cls(
            run=_section("run", RunSection.from_mapping, RunSection()),
            model=_section("model", ModelSection.from_mapping, ModelSection()),
            data=_section("data", DataSection.from_mapping, DataSection()),
            engine=_section("engine", EngineSection.from_mapping, EngineSection()),
            optim=_section("optim", OptimSection.from_mapping, OptimSection()),
            schedule=_section("schedule", ScheduleSection.from_mapping, ScheduleSection()),
            tracking=_section("tracking", TrackingSection.from_mapping, TrackingSection()),
            eval=_section("eval", EvalSection.from_mapping, EvalSection()),
        )

    def validate(self) -> None:
        """Core-level structural validation.

        Validates common boundaries (stage, backend, strategy, required
        fields).  Model-specific and dataset-specific validation is
        delegated to plugins (preflight, not hot-path).
        """
        # Run
        if self.run.stage not in _ALLOWED_STAGES:
            raise ValueError(
                f"run.stage must be one of {_ALLOWED_STAGES}, got {self.run.stage!r}"
            )

        # Model
        if self.run.stage in ("train", "test") and not self.model.type:
            raise ValueError("model.type is required for train/test")

        # Engine
        if self.engine.backend not in _ALLOWED_BACKENDS:
            raise ValueError(
                f"engine.backend must be one of {_ALLOWED_BACKENDS}, got {self.engine.backend!r}"
            )
        if self.engine.strategy not in _ALLOWED_STRATEGIES:
            raise ValueError(
                f"engine.strategy must be one of {_ALLOWED_STRATEGIES}, got {self.engine.strategy!r}"
            )

        # Data (train requires a dataset)
        if self.run.stage == "train" and not self.data.train.dataset_name:
            raise ValueError("data.train.dataset_name is required for training")

        # Eval
        if self.run.stage == "test" and not self.eval.metrics:
            raise ValueError("eval.metrics is required for test stage")

        # Schedule: steps must be > 0 for training
        if self.run.stage == "train" and self.schedule.steps <= 0:
            raise ValueError("schedule.steps must be > 0 for training")

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict summary (for logging / manifest)."""
        return {
            "run": {
                "name": self.run.name,
                "stage": self.run.stage,
                "seed": self.run.seed,
                "output_dir": self.run.output_dir,
            },
            "model": {
                "type": self.model.type,
            },
            "data": {
                "train_dataset": self.data.train.dataset_name,
                "val_dataset": self.data.val.dataset_name,
            },
            "engine": {
                "backend": self.engine.backend,
                "strategy": self.engine.strategy,
            },
            "optim": {
                "type": self.optim.type,
            },
            "schedule": {
                "type": self.schedule.type,
                "steps": self.schedule.steps,
                "base_lr": self.schedule.base_lr,
            },
        }