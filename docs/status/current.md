# 当前工程状态

> **ElysiaCV v0.1.0（Alpha）** · 最近代码核对：2026-10-12 · 以当前 `main` 分支的实际文件与实现为依据。

ElysiaCV 正在建设面向图像分类、目标检测和语义分割的统一研究框架。当前已形成模型插件、Registry、配置系统、训练循环、分布式策略等基础模块；**模块存在不等于 CLI、多任务数据和标准评测已经完成端到端闭环。**

## 已有实现

| 领域 | 当前代码 |
| --- | --- |
| **Plugin / Registry** | `core/plugin.py`、`core/registry.py` 和 `models/registry.py`；基于目录发现模型插件。 |
| **配置系统** | YAML `_base_` 继承、环境变量展开、配置对象、参数校验和 `TrainingPlan`。 |
| **模型与组件** | 分类 ResNet / Cassava、检测结构示例、分割 FCN；Backbone、Neck、Head、Loss、Metric 等。 |
| **训练模块** | `training/engine.py`、`loop.py`、HookBus、Checkpoint、DDP、MetricsRuntime、Optimizer / Scheduler。 |
| **测试代码** | `tests/` 包含配置、插件、分类、检测、分割及训练逻辑测试源码；目前未核实全套测试在当前分支通过。 |
| **运行入口** | `run_experiment.sh` 提供 `--help` 和 `--print-command`；后者只输出命令、不启动训练。 |

## 尚未闭环 / 已发现问题

| 位置 | 当前情况 | 影响 |
| --- | --- | --- |
| **数据层** | 当前 `main` 树中不存在 `src/elysia_cv/data/`，但 `elysia_cv/__init__.py`、CLI 和 Engine 都引用它。 | Python 包导入及常规训练 / 测试入口被阻断。 |
| **训练 CLI** | `cli.py` 中的训练入口直接使用 Cassava Runtime；训练分支还引用未定义的 `save_freq`。 | 无法视作已完成的统一多任务训练 CLI。 |
| **Test / Infer** | `_cmd_test`、`_cmd_infer` 当前输出占位提示。 | 尚不能执行正式评测和图像推理。 |
| **Detection Runtime** | 当前检测训练 Loss 使用随机伪类别与随机框目标。 | 用于说明前向和 Loss 结构，不构成完整 Faster R-CNN 检测算法或 COCO 训练。 |
| **Detection Metric** | `MeanAP` 返回的是 IoU 近似占位结果。 | 不能把该值作为标准 COCO mAP。 |
| **真实任务验证** | 仓库包含 ImageNet / COCO / ADE20K 配置和相应模块，但未看到对齐的数据加载与端到端复现证据。 | 真实训练、复现和性能结果需后续单独验证。 |

## 已登记插件与示例

| 方向 | 对应实现 |
| --- | --- |
| 分类 | `resnet` 与 `cassava`（MLP / SimpleCNN / ResNet / ViT） |
| 检测 | `faster_rcnn`（ResNet + FPN + DetHead 的结构 / 代理损失示例） |
| 分割 | `fcn`（ResNet + IdentityNeck + SegHead） |

插件清单指源码中存在的 `plugin.py`、`runtime.py` 和 ModelSpec，并非统一训练入口已认证列表。

## 下一步

1. **P0 · Foundation**：补齐数据模块与包导入链路；修复训练 CLI 阻断点；建立最小可执行验证。
2. **P1 · Task closure**：真实 Dataset / Transform 接入，训练主路径按 Plugin Runtime 派发，分别闭环分类、检测、分割。
3. **P2 · Evaluation**：实现 Test / Infer，补齐检测匹配与标准评测口径，核验 Checkpoint 生命周期。
4. **P3 · Expansion**：完成基础能力后，再按需求扩展新的检测、分割模型与数据源。

## 相关资料

- [项目介绍与快速开始](../../README.md)
- [架构合同](../infrastructure/architecture.md)
- [训练指南](../guides/training.md)
- [扩展指南](../guides/extension.md)

> 本页是代码对照后的状态记录，不将历史文档中的计划性描述或独立模块测试自动视为完整工程验收。
