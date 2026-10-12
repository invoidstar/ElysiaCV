# 当前工程状态

> **ElysiaCV v0.1.0（Alpha）** · 最近核对：2026-10-12 · 以 `main` 分支实际代码为依据。

## ✅ 已完成

| 模块 | 状态 |
|------|------|
| **Plugin / Registry** | `core/plugin.py`、`core/registry.py`、`models/registry.py`；目录扫描自发现 |
| **配置系统** | YAML `_base_` 多源继承、环境变量展开、dataclass 配置对象 |
| **配置拆分** | `configs/{models,datasets,experiments}/` 模型+数据集分离组合式配置 |
| **Backbone 组件** | MLP、SimpleCNN、ResNet(18/34/50)、ViT(B/16, L/16) |
| **训练引擎** | `TrainingEngine` → `TrainerLoop`；梯度累积 + DDP no_sync |
| **Hook 系统** | HookBus + Logger(TTY-frzwe tqdm/非TTY 百分比) + Checkpoint + Eval + LR |
| **DDP 支持** | `torch.distributed` init + `wrap_ddp` + `gradient_sync_context` |
| **CLI 统一** | `cli.py` 使用 `TrainingEngine`；模型+数据通过回调注入，不再硬编码 |
| **数据层** | `src/elysia_cv/data/` 已纳入版本控制；`CassavaDataset` + transforms |
| **分类实验** | Cassava 7 模型对比：MLP(63%), SimpleCNN(81%), ResNet-18(83%) |
| **分析工具** | `tools/analyze.py`：混淆矩阵 + GradCAM 热力图 |
| **项目分离** | `projects/cassava/`：scripts + tools + DDP launcher |

## ⚠️ 尚未闭环

| 位置 | 当前情况 |
|------|---------|
| **Test / Infer 命令** | `_cmd_test`、`_cmd_infer` 输出占位提示 |
| **Detection Runtime** | 使用随机伪标签与随机框目标（结构演示，非可训练算法） |
| **Detection Metric** | `MeanAP` 为占位实现 |
| **COCO / ADE20K 数据集** | 通用 `ImageFolderDataset` 可用，但无 COCO JSON / ADE20K 解析 |
| **端到端检测/分割训练** | 配置已就位，runtime 缺失真实 loss/assignment |

## 下一步

1. **P1**：实现 test/infer CLI，接入 validation pipeline
2. **P2**：真实 Detection Runtime（anchor + IoU + NMS）
3. **P3**：COCO / ADE20K Dataset 类实现

## 相关资料

- [架构合同](../infrastructure/architecture.md)
- [训练指南](../guides/training.md)
- [扩展指南](../guides/extension.md)