"""Training plan — compiles CVConfig into an immutable, fully-resolved plan dict.

Adapted from ElysiaRobot's native_plan.py. The plan is a serializable
snapshot of ALL training configuration that the engine needs, with all
defaults resolved and cross-section dependencies validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from elysia_cv.config.config import CVConfig
from elysia_cv.config.validation import validate_cross_section


@dataclass(frozen=True)
class TrainingPlan:
    """Fully resolved, immutable training plan derived from CVConfig."""

    run: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    engine: dict[str, Any] = field(default_factory=dict)
    optim: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    tracking: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: CVConfig) -> "TrainingPlan":
        """Compile a CVConfig into a training plan."""
        validate_cross_section(config)

        return cls(
            run={
                "name": config.run.name,
                "stage": config.run.stage,
                "seed": config.run.seed,
                "output_dir": config.run.output_dir,
                "test_run": config.run.test_run,
            },
            model={
                "type": config.model.type,
                "pretrained_path": config.model.pretrained_path,
                "backbone": dict(config.model.backbone),
                "neck": dict(config.model.neck),
                "head": dict(config.model.head),
                "kwargs": dict(config.model.kwargs),
            },
            data={
                "train": {
                    "dataset_name": config.data.train.dataset_name,
                    "dataset_root": config.data.train.dataset_root,
                    "batch_size": config.data.train.batch_size,
                    "num_workers": config.data.train.num_workers,
                    "transforms": list(config.data.train.transforms),
                },
                "val": {
                    "dataset_name": config.data.val.dataset_name,
                    "dataset_root": config.data.val.dataset_root,
                    "batch_size": config.data.val.batch_size,
                    "num_workers": config.data.val.num_workers,
                    "transforms": list(config.data.val.transforms),
                },
            },
            engine={
                "backend": config.engine.backend,
                "strategy": config.engine.strategy,
                "distributed": {
                    "proc_per_node": config.engine.distributed.proc_per_node,
                    "find_unused_parameters": config.engine.distributed.ddp_find_unused_parameters,
                    "broadcast_buffers": config.engine.distributed.ddp_broadcast_buffers,
                    "bucket_cap_mb": config.engine.distributed.ddp_bucket_cap_mb,
                    "gradient_as_bucket_view": config.engine.distributed.ddp_gradient_as_bucket_view,
                    "static_graph": config.engine.distributed.ddp_static_graph,
                },
                "precision": {
                    "mixed_precision": config.engine.precision.mixed_precision,
                    "dtype": config.engine.precision.dtype,
                    "cudnn_benchmark": config.engine.precision.cudnn_benchmark,
                    "allow_tf32": config.engine.precision.allow_tf32,
                },
                "compile": {
                    "enabled": config.engine.compile.enabled,
                    "mode": config.engine.compile.mode,
                },
                "optimization": {
                    "gradient_accumulation_steps": config.engine.optimization.gradient_accumulation_steps,
                    "grad_clip_norm": config.engine.optimization.grad_clip_norm,
                    "zero_grad_set_to_none": config.engine.optimization.optimizer_zero_grad_set_to_none,
                },
            },
            optim={
                "type": config.optim.type,
                "params": dict(config.optim.params),
            },
            schedule={
                "type": config.schedule.type,
                "steps": config.schedule.steps,
                "base_lr": config.schedule.base_lr,
                "min_lr": config.schedule.min_lr,
                "warmup_steps": config.schedule.warmup_steps,
                "params": dict(config.schedule.params),
            },
            tracking={
                "log_freq": config.tracking.log_freq,
                "wandb_enabled": config.tracking.wandb.enabled,
                "wandb_project": config.tracking.wandb.project,
            },
            eval={
                "eval_freq": config.eval.eval_freq,
                "metrics": list(config.eval.metrics),
                "save_best": config.eval.save_best,
                "save_final": config.eval.save_final,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "model": self.model,
            "data": self.data,
            "engine": self.engine,
            "optim": self.optim,
            "schedule": self.schedule,
            "tracking": self.tracking,
            "eval": self.eval,
        }