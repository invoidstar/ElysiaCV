# AGENTS.md — training/

> 本模块是训练基础设施：训练循环、Hook 系统、分布式策略。

---

## 关键文件

| 文件 | 作用 | 状态 |
|------|------|------|
| `loop.py` | `TrainerLoop` — 主训练循环 | ✅ CLI 已全面使用 |
| `engine.py` | `TrainingEngine` — 入口 + 模型/数据构建 | 被 cli.py 调用 |
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

**当前状态**：✅ CLI 训练已通过 HookBus 驱动。`TrainerLoop` 内置了 LoggerHook、CheckpointHook、EvalHook 等，`engine.py` 统一管理模型/数据构建（通过 `model_builder.py` 共享逻辑）。

---

## 依赖规则

```
training/ → core/ + config/ + models/ (通过 import)
training/ ✗→ 不被 models/ 或 components/ reverse import
```