# AGENTS.md — training/

> 本模块是训练基础设施：训练循环、Hook 系统、分布式策略。

---

## 关键文件

| 文件 | 作用 | 状态 |
|------|------|------|
| `loop.py` | `TrainerLoop` — 主训练循环 | ⚠️ CLI 暂未使用（使用直接训练循环替代） |
| `engine.py` | `TrainingEngine` — step 级逻辑 | 被 loop.py 调用 |
| `plan.py` | `TrainingPlan` — 训练前校验 | 每次训练前执行 |
| `strategy.py` | `ExecutionStrategy` — grad_accum/grad_clip 配置 | 被 cli.py 使用 |
| `distributed.py` | DDP/FSDP 支持 | 多卡训练时激活 |
| `hooks/` | HookBus + 内置 Hook | checkpoint、logger、evaluation |

---

## Hook 系统

```python
from elysia_cv.training.hooks.base import HookBus
from elysia_cv.training.hooks.checkpoint import CheckpointHook
from elysia_cv.training.hooks.logger import LoggerHook

hooks = HookBus()
hooks.register(LoggerHook(log_freq=50))
hooks.register(CheckpointHook(output_dir="./outputs/checkpoints", save_freq=1000))
```

**当前状态**：CLI 中的训练循环未使用 HookBus（接口不匹配导致），而是直接在 `cli.py` 中内联了训练循环。未来计划重构为通过 HookBus 驱动。

---

## 依赖规则

```
training/ → core/ + config/ + models/ (通过 import)
training/ ✗→ 不被 models/ 或 components/ reverse import
```