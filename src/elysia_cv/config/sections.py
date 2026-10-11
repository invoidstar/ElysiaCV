"""Configuration section dataclasses — the typed schema for ElysiaCV configs.

Each section corresponds to a top-level YAML key:
  model | data | engine | tracking | optim | schedule | eval | run

Inspired by ElysiaRobot's configs/sections.py but tailored for CV tasks.
All dataclasses are frozen and provide ``from_mapping()`` constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Run section
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSection:
    """Top-level run metadata."""

    name: str = ""
    stage: str = "train"  # "train" | "test" | "infer"
    seed: int = 42
    output_dir: str = "./work_dirs"
    test_run: bool = False

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RunSection":
        return cls(
            name=str(mapping.get("name", "")),
            stage=str(mapping.get("stage", "train")),
            seed=int(mapping.get("seed", 42)),
            output_dir=str(mapping.get("output_dir", "./work_dirs")),
            test_run=bool(mapping.get("test_run", False)),
        )


# ---------------------------------------------------------------------------
# Model section — mirrors ElysiaRobot's NativePolicyConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSection:
    """Model configuration section.

    The ``type`` field selects a registered :class:`ModelPlugin`.
    ``pretrained_path`` overrides / adds pre-trained weights.
    ``backbone``, ``neck``, ``head`` are optional overrides for shared components.
    ``kwargs`` is a passthrough dict forwarded to the model builder.
    """

    type: str = ""
    pretrained_path: str | None = None
    backbone: dict[str, Any] = field(default_factory=dict)
    neck: dict[str, Any] = field(default_factory=dict)
    head: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ModelSection":
        return cls(
            type=str(mapping.get("type", "")),
            pretrained_path=_optional_str(mapping.get("pretrained_path")),
            backbone=dict(mapping.get("backbone") or {}),
            neck=dict(mapping.get("neck") or {}),
            head=dict(mapping.get("head") or {}),
            kwargs=dict(mapping.get("kwargs") or {}),
        )


# ---------------------------------------------------------------------------
# Data section
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSection:
    """Data pipeline configuration.

    ``train`` / ``val`` / ``test`` each hold sub-configs for the dataset,
    loader, and transforms used in that phase.  ``transforms`` is a list
    of ``{"type": "Resize", "size": (224, 224)}`` dicts.
    """

    train: _DataSubSection = field(default_factory=lambda: _DataSubSection())
    val: _DataSubSection = field(default_factory=lambda: _DataSubSection())
    test: _DataSubSection = field(default_factory=lambda: _DataSubSection())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DataSection":
        return cls(
            train=_DataSubSection.from_mapping(mapping.get("train") or {}),
            val=_DataSubSection.from_mapping(mapping.get("val") or {}),
            test=_DataSubSection.from_mapping(mapping.get("test") or {}),
        )


@dataclass(frozen=True)
class _DataSubSection:
    dataset_name: str = ""
    dataset_root: str = ""
    split: str = ""
    batch_size: int = 32
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    transforms: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "_DataSubSection":
        loader = mapping.get("loader") or {}
        return cls(
            dataset_name=str(mapping.get("dataset_name", "")),
            dataset_root=str(mapping.get("dataset_root", "")),
            split=str(mapping.get("split", "")),
            batch_size=int(loader.get("batch_size", 32)),
            num_workers=int(loader.get("num_workers", 2)),
            prefetch_factor=int(loader.get("prefetch_factor", 2)),
            pin_memory=bool(loader.get("pin_memory", True)),
            persistent_workers=bool(loader.get("persistent_workers", True)),
            drop_last=bool(loader.get("drop_last", False)),
            transforms=list(mapping.get("transforms") or []),
        )


# ---------------------------------------------------------------------------
# Engine section — mirrors ElysiaRobot's NativeEngineConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineSection:
    """Training engine configuration.

    Sub-sections:
      ``distributed`` — DDP / FSDP settings.
      ``precision`` — mixed-precision (amp, bf16, fp32).
      ``compile`` — torch.compile toggle.
      ``optimization`` — gradient accumulation, clip, zero_grad hints.
    """

    backend: str = "native"       # "native" | "accelerate"
    strategy: str = "ddp"         # "ddp" | "fsdp" | "single"
    distributed: _DistributedSubSection = field(default_factory=lambda: _DistributedSubSection())
    precision: _PrecisionSubSection = field(default_factory=lambda: _PrecisionSubSection())
    compile: _CompileSubSection = field(default_factory=lambda: _CompileSubSection())
    optimization: _OptSubSection = field(default_factory=lambda: _OptSubSection())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "EngineSection":
        return cls(
            backend=str(mapping.get("backend", "native")),
            strategy=str(mapping.get("strategy", "ddp")),
            distributed=_DistributedSubSection.from_mapping(mapping.get("distributed") or {}),
            precision=_PrecisionSubSection.from_mapping(mapping.get("precision") or {}),
            compile=_CompileSubSection.from_mapping(mapping.get("compile") or {}),
            optimization=_OptSubSection.from_mapping(mapping.get("optimization") or {}),
        )


@dataclass(frozen=True)
class _DistributedSubSection:
    proc_per_node: int = 1
    node_count: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    ddp_find_unused_parameters: bool = True
    ddp_broadcast_buffers: bool = False
    ddp_bucket_cap_mb: int = 64
    ddp_gradient_as_bucket_view: bool = True
    ddp_static_graph: bool = False
    fsdp_enable: bool = False

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "_DistributedSubSection":
        return cls(
            proc_per_node=int(mapping.get("proc_per_node", 1)),
            node_count=int(mapping.get("node_count", 1)),
            master_addr=str(mapping.get("master_addr", "127.0.0.1")),
            master_port=int(mapping.get("master_port", 29500)),
            ddp_find_unused_parameters=bool(mapping.get("ddp_find_unused_parameters", True)),
            ddp_broadcast_buffers=bool(mapping.get("ddp_broadcast_buffers", False)),
            ddp_bucket_cap_mb=int(mapping.get("ddp_bucket_cap_mb", 64)),
            ddp_gradient_as_bucket_view=bool(mapping.get("ddp_gradient_as_bucket_view", True)),
            ddp_static_graph=bool(mapping.get("ddp_static_graph", False)),
            fsdp_enable=bool(mapping.get("fsdp_enable", False)),
        )


@dataclass(frozen=True)
class _PrecisionSubSection:
    mixed_precision: bool = True
    dtype: str = "bfloat16"       # "bfloat16" | "float16"
    cudnn_benchmark: bool = True
    allow_tf32: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "_PrecisionSubSection":
        return cls(
            mixed_precision=bool(mapping.get("mixed_precision", True)),
            dtype=str(mapping.get("dtype", "bfloat16")),
            cudnn_benchmark=bool(mapping.get("cudnn_benchmark", True)),
            allow_tf32=bool(mapping.get("allow_tf32", True)),
        )


@dataclass(frozen=True)
class _CompileSubSection:
    enabled: bool = False
    mode: str = "reduce-overhead"  # "default" | "reduce-overhead" | "max-autotune"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "_CompileSubSection":
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            mode=str(mapping.get("mode", "reduce-overhead")),
        )


@dataclass(frozen=True)
class _OptSubSection:
    gradient_accumulation_steps: int = 1
    grad_clip_norm: float | None = None
    optimizer_zero_grad_set_to_none: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "_OptSubSection":
        return cls(
            gradient_accumulation_steps=int(mapping.get("gradient_accumulation_steps", 1)),
            grad_clip_norm=_optional_float(mapping.get("grad_clip_norm")),
            optimizer_zero_grad_set_to_none=bool(mapping.get("optimizer_zero_grad_set_to_none", True)),
        )


# ---------------------------------------------------------------------------
# Optim section
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimSection:
    """Optimizer configuration — passed to ``build_optimizer()``.

    ``type`` selects from ``OPTIMIZERS`` registry (SGD, AdamW, ...).
    ``params`` is forwarded as kwargs.
    """

    type: str = "AdamW"
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "OptimSection":
        return cls(
            type=str(mapping.get("type", "AdamW")),
            params=dict(mapping.get("params") or {}),
        )


# ---------------------------------------------------------------------------
# Schedule section
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleSection:
    """Learning-rate schedule configuration.

    ``type`` selects from ``SCHEDULERS`` registry.
    ``steps`` is the total number of steps (shared with run config).
    """

    type: str = "CosineAnnealing"
    steps: int = 100000
    warmup_steps: int = 1000
    base_lr: float = 0.001
    min_lr: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScheduleSection":
        return cls(
            type=str(mapping.get("type", "CosineAnnealing")),
            steps=int(mapping.get("steps", 100000)),
            warmup_steps=int(mapping.get("warmup_steps", 1000)),
            base_lr=float(mapping.get("base_lr", 0.001)),
            min_lr=float(mapping.get("min_lr", 0.0)),
            params=dict(mapping.get("params") or {}),
        )


# ---------------------------------------------------------------------------
# Tracking section — mirrors ElysiaRobot's NativeTrackingConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackingSection:
    """Logging / tracking configuration."""

    log_freq: int = 50
    wandb: _WandbSubSection = field(default_factory=lambda: _WandbSubSection())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrackingSection":
        return cls(
            log_freq=int(mapping.get("log_freq", 50)),
            wandb=_WandbSubSection.from_mapping(mapping.get("wandb") or {}),
        )


@dataclass(frozen=True)
class _WandbSubSection:
    enabled: bool = False
    project: str = "ElysiaCV"
    entity: str | None = None
    run_id: str | None = None
    resume: bool = False
    mode: str = "online"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "_WandbSubSection":
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            project=str(mapping.get("project", "ElysiaCV")),
            entity=_optional_str(mapping.get("entity")),
            run_id=_optional_str(mapping.get("run_id")),
            resume=bool(mapping.get("resume", False)),
            mode=str(mapping.get("mode", "online")),
        )


# ---------------------------------------------------------------------------
# Eval section
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSection:
    """Evaluation configuration."""

    eval_freq: int = 1000
    metrics: list[str] = field(default_factory=lambda: ["Accuracy"])
    save_best: str | None = None   # metric name for best-model selection
    save_final: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "EvalSection":
        return cls(
            eval_freq=int(mapping.get("eval_freq", 1000)),
            metrics=list(mapping.get("metrics", ["Accuracy"])),
            save_best=_optional_str(mapping.get("save_best")),
            save_final=bool(mapping.get("save_final", True)),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _optional_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _optional_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None