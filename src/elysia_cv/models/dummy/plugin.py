"""Dummy model plugin — example / test fixture demonstrating the contract.

This minimal plugin declares one model variant and exits; no actual model
code is loaded until runtime.  Used to validate the discovery infrastructure.
"""

from __future__ import annotations

from elysia_cv.core.plugin import SimpleModelPlugin

MODEL_PLUGIN = SimpleModelPlugin(
    name="dummy",
    task_type="classification",
    model_specs=[
        {"name": "dummy_v1", "backbone": "dummy", "num_params": 0},
    ],
    readiness_level="metadata_only",
)