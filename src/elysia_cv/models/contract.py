"""Model Plugin contract — adapted from ElysiaRobot policies/contract.py.

Defines the full contract that every model plugin must satisfy.
A model plugin declares its identity, task type, and all factories
needed by the training engine: model builder, runtime, data pipeline,
optimizer specs, and checkpoint codec.

The contract is intentionally **maximal**: a plugin may choose to
implement only the methods it needs.  Missing methods are detected
at plan-compile time (preflight), not at runtime.

This is the CV equivalent of ``TrainablePolicyPlugin``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ModelRuntimeProtocol(Protocol):
    """Protocol for model training/eval runtimes.

    The runtime is the bridge between the training engine and the
    model implementation.  It knows how to build the model, run a
    training forward pass, and run evaluation.
    """

    def build_model(self, cfg: dict[str, Any]) -> Any:
        """Build and return the model from config."""

    def forward_train(self, model: Any, batch: dict[str, Any]) -> dict[str, Any]:
        """Run one training forward pass.

        Must return a dict with at least ``loss`` (the scalar for backward).
        May include additional per-objective losses and metrics.
        """

    def forward_test(self, model: Any, batch: dict[str, Any]) -> dict[str, Any]:
        """Run one evaluation forward pass and return predictions."""

    def get_default_config(self) -> dict[str, Any]:
        """Return sensible default config for this model."""

    def validate_config(self, cfg: dict[str, Any]) -> list[str]:
        """Validate a config dict; return list of error messages (empty = OK)."""


@runtime_checkable
class ModelPlugin(Protocol):
    """Full model plugin contract.

    Plugin modules expose ``MODEL_PLUGIN`` (an object satisfying this
    protocol).  The Core only reads lightweight metadata; factory
    methods are called by the plan compiler / engine when needed.
    """

    name: str
    task_type: str
    api_version: str
    readiness_level: str

    def model_specs(self) -> list[dict[str, Any]]:
        """Return metadata specs describing available model variants."""

    def get_runtime_target(self) -> str:
        """Return ``module:Class`` string for the runtime class."""

    def get_model_build_target(self) -> str:
        """Return ``module:func`` string for the model builder."""

    def get_default_config(self) -> dict[str, Any]:
        """Return default training config for this model."""

    def validate_config(self, cfg: dict[str, Any]) -> list[str]:
        """Validate a training config; return error list."""


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing one model variant."""

    name: str
    task_type: str
    backbone: str | None = None
    neck: str | None = None
    head: str | None = None
    input_size: tuple[int, int] = (224, 224)
    num_params: int | None = None
    flops: int | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "task_type": self.task_type,
            "backbone": self.backbone,
            "neck": self.neck,
            "head": self.head,
            "input_size": list(self.input_size),
        }
        if self.num_params is not None:
            d["num_params"] = self.num_params
        if self.flops is not None:
            d["flops"] = self.flops
        if self.extra:
            d.update(self.extra)
        return d