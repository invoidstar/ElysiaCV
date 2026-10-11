# 架构合同

> 状态：Active（长期有效） | 最后更新：2026-10

本文档定义 ElysiaCV 的模块职责边界、依赖关系和核心数据流。**不描述具体实现**，只写合同。

---

## 全局数据流

```
User Config (YAML)
    │
    ▼
loader.py                         # _base_ 继承 + env 展开
    │
    ▼
CVConfig (frozen dataclass)       # 类型化配置
    │
    ▼
TrainingPlan                      # 不可变 resolved plan
    │
    ├──→ ModelRegistry.discover() # 自动扫描 models/*/plugin.py
    ├──→ BACKBONES/NECKS/HEADS    # 从 registry 构建组件
    ├──→ Dataset + Transform      # 数据流水线
    └──→ Optimizer + Scheduler    # 优化器组合
    │
    ▼
TrainingEngine
    │
    ├── Model = build_model(plan)         # backbone → neck → head
    ├── Dataloader = build_dataloader()   # dataset + sampler + transforms
    ├── Optimizer + Scheduler
    ├── HookBus (checkpoint / logger / eval / lr)
    │
    ▼
TrainerLoop
    │
    ├── for step in range(max_steps):
    │     ├── micro_loop (grad_accum):
    │     │     ├── with model.no_sync() if not is_sync:
    │     │     │     forward_train(model, batch) → loss / backward
    │     │     └── if is_sync: clip_grad → optimizer.step() → scheduler.step()
    │     │
    │     ├── hook.after_step (log / checkpoint / eval)
    │     └── hook.after_epoch
    │
    └── hook.after_run
```

---

## 模块边界

### `core/` — 合同与基础设施

| 模块 | 唯一职责 | 不做什么 |
|------|---------|---------|
| `plugin.py` | 定义 Plugin 协议 + 自动发现逻辑 | 不导入模型代码 |
| `registry.py` | name→class 映射 + `build_from_cfg` | 不预设注册内容 |
| `errors.py` | 异常体系 | 不包含业务逻辑 |
| `capabilities.py` | 能力枚举 | 不包含实现 |

**依赖方向**：`core/` 被所有模块依赖，但不依赖任何模块。不导入 torch。

### `config/` — 配置系统

| 模块 | 唯一职责 |
|------|---------|
| `loader.py` | YAML 加载 + `_base_` + override + env |
| `sections.py` | 配置节 typed dataclass |
| `validation.py` | 跨节一致性校验 |

**依赖方向**：`config/` → `core/`。不被 `models/` 或 `components/` 依赖。

### `data/` — 数据流水线

| 模块 | 唯一职责 |
|------|---------|
| `dataset.py` | Map-style Dataset 基类 |
| `dataloader.py` | DDP-aware DataLoader builder |
| `sampler.py` | DistributedSampler |
| `transforms/` | 注册式 Transform pipeline |

**依赖方向**：`data/` → `core/` (Registry)。不依赖 `models/` 或 `training/`。

### `models/` — 模型插件

| 目录 | 唯一职责 |
|------|---------|
| `contract.py` | ModelPlugin 完整契约 (Protocol + ModelSpec) |
| `registry.py` | 单例 ModelRegistry (懒加载自动发现) |
| `{task}/{name}/plugin.py` | MODEL_PLUGIN — 元数据 |
| `{task}/{name}/runtime.py` | build_model / forward_train / forward_test |

**依赖方向**：`models/` → `core/` + `components/` + `losses/`。不被 `training/` 依赖（engine 通过 runtime 函数间接调用）。

### `components/` — 共享零件

| 目录 | 职责 |
|------|------|
| `backbones/` | 视觉骨干网络 |
| `necks/` | 特征融合层 |
| `heads/` | 任务头 |

所有零件通过全局 Registry 注册，被 `models/*/runtime.py` 的 `build_model()` 调用。

### `training/` — 训练引擎

| 模块 | 唯一职责 |
|------|---------|
| `engine.py` | 顶层生命周期 (build → run → cleanup) |
| `loop.py` | 设备无关训练循环 |
| `strategy.py` | 梯度累积 + DDP sync 决策 |
| `distributed.py` | DDP init / wrap / all-reduce |
| `hooks/base.py` | Hook 事件总线 |
| `hooks/checkpoint.py` | 保存 / 恢复 |
| `hooks/logger.py` | 日志输出 |
| `metrics_runtime.py` | 设备端指标累积 |

**依赖方向**：`training/` → `core/` + `models/` (弱依赖，通过函数对象)。不依赖 `data/` 或 `components/`。

---

## 所有权约定

| 概念 | 所有者 | 说明 |
|------|--------|------|
| 训练循环控制 | `training/loop.py` | 决定何时 forward/backward/step |
| 前向计算逻辑 | `models/*/runtime.py` | 决定如何计算 loss + metrics |
| 配置管理 | `config/` | 唯一事实来源 |
| 模型发现 | `core/plugin.py` + `models/registry.py` | 自动扫描无注册中心 |
| 零件组装 | `components/` + Registry | 注册式 builder pattern |