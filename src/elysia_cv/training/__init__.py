"""Training infrastructure — engine, loop, hooks, distributed, checkpoint.

Public API:
  - ``TrainingEngine`` — top-level lifecycle (build → train → cleanup)
  - ``TrainerLoop`` — device/backend-agnostic training loop
  - ``ExecutionStrategy`` — gradient accumulation + DDP sync control
  - ``TrainingPlan`` — immutable resolved training config
  - ``Hook``, ``HookBus`` — event system for checkpoints/logging/eval
  - ``DistributedContext`` — DDP runtime state
  - ``LoopState``, ``LoopContext`` — training loop state/context
"""

from __future__ import annotations