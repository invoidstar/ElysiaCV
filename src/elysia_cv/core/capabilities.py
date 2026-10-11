"""Core capability names — adapted from ElysiaRobot for CV tasks.

A capability is a stable string that a plugin declares to the Core.
It describes *what the plugin can provide*, not how the model trains
or what architecture it uses.
"""

from __future__ import annotations

from enum import Enum


class CoreCapability(str, Enum):
    """Stable capabilities a plugin may declare."""

    MODEL_SPEC = "model_spec"                     # declares model architecture metadata
    NATIVE_RUNTIME = "native_runtime"              # provides training/eval runtime
    STATIC_PREFLIGHT = "static_preflight"         # supports dry-run validation
    DATASET_SCHEMA = "dataset_schema"             # declares dataset schema/format
    AUGMENTATION = "augmentation"                 # provides custom augmentations
    PRETRAINED_WEIGHTS = "pretrained_weights"     # provides pre-trained checkpoint
    METRIC = "metric"                              # provides evaluation metric(s)


MODEL_PLUGIN_CAPABILITIES = (
    CoreCapability.MODEL_SPEC.value,
    CoreCapability.NATIVE_RUNTIME.value,
)

DATASET_PLUGIN_CAPABILITIES = (CoreCapability.DATASET_SCHEMA.value,)

METRIC_PLUGIN_CAPABILITIES = (CoreCapability.METRIC.value,)