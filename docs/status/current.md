# 当前工程状态

> **ElysiaCV v0.2.0（Alpha）** · 最近核对：2026-10-20 · 以 `main` 分支实际代码为依据。

## ✅ 已完成

| 模块 | 状态 |
|------|------|
| **Plugin / Registry** | `core/plugin.py`、`core/registry.py`、`models/registry.py`；目录扫描自发现 |
| **配置系统** | YAML `_base_` 多源继承、环境变量展开、dataclass 配置对象、`elysia config dump` 完整展开 |
| **配置拆分** | `configs/{models,datasets,experiments}/` 模型+数据集分离组合式配置 |
| **Backbone 组件** | MLP、SimpleCNN、ResNet(18/34/50/101)、ViT(B/16, L/16) |
| **Necks & Heads** | FPN、IdentityNeck、LinearClsHead、ClsHead、DetHead、SegHead |
| **训练引擎** | `TrainingEngine` → `TrainerLoop`；梯度累积 + DDP/FSDP no_sync；`--resume` 断点续训 |
| **Hook 系统** | HookBus + Logger(EMA平滑/TTY tqdm/非TTY百分比) + Checkpoint + Eval + LR |
| **DDP / FSDP** | `torch.distributed` init + `wrap_ddp` / `wrap_fsdp` + `gradient_sync_context` |
| **CLI 统一** | `train`/`test`/`infer`/`config dump` 四命令；模型+数据通过回调注入 |
| **数据层** | `src/elysia_cv/data/`：Compose pipeline、COCO/ImageFolder/Cassava Dataset、DataPrefetcher 异步预取、Data Adapter（HF/torchvision） |
| **变换管线** | `Compose` 类（`__repr__` 可视化）+ LoadImageFromFile/LoadAnnotations/Resize/Normalize/PackDetInputs/PackSegInputs |
| **损失函数** | CrossEntropy、FocalLoss、SmoothL1、IoU/GIoU、DiceLoss |
| **分类** | Cassava 9 模型：MLP(63%)、SimpleCNN(81%)、ResNet-18/34/50(83%+)、ViT-B/16、ViT-L/16、Swin-T/B |
| **检测 (FCOS)** | FCOS-style anchor-free：ResNet+FPN+DetHead、center-sampling 真损失、NMS 后处理、真 COCO mAP |
| **分割 (FCN)** | ResNet-50/101 + SegHead，CE/Dice 损失，MeanIoU |
| **评测指标** | Accuracy、TopK、MeanAP (COCO-standard 101-pt mAP/mAP_50/mAP_75)、MeanIoU |
| **优化器** | SGD、Adam、AdamW、RMSprop + MultiStepLR/CosineAnnealing/StepLR/ReduceLROnPlateau/ConstantLR + Warmup |
| **分析工具** | `tools/analyze.py`：混淆矩阵 + GradCAM 热力图 |
| **可视化** | `utils/visualization.py`：`draw_bboxes()` + `draw_segmentation_overlay()` |
| **MessageHub** | 全局标量状态总线：`update_scalar`/`get_all_scalars`/best-value 追踪 |
| **项目分离** | `projects/HERBS/`：scripts + tools + DDP launcher |

## ⚠️ 尚未闭环

| 位置 | 当前情况 |
|------|---------|
| **COCO 全量训练验证** | 需要 COCO 2017 数据集在 `/data/coco/` 下做端到端训练验证 |
| **模型动物园** | 无预训练权重发布；backbone 依赖 torchvision 内置权重 |

## 已完成路线图

| 阶段 | 重点 | 状态 |
|------|------|------|
| **Phase 1 · 核心闭环** | AMP 混合精度、`test`/`infer` CLI、METRICS 注册表、cassava 解耦 | ✅ |
| **Phase 2 · 多任务支持** | COCO 数据加载、FCOS 真检测训练、真 MeanAP、分类/检测/分割闭环 | ✅ |
| **Phase 3 · 开发者体验** | `--resume` 断点续训、MessageHub、EMA 日志、DataPrefetcher、`config dump`、`set_epoch()` | ✅ |
| **Phase 4 · 质量修补** | fcos 重命名、detection config 校验、Data Adapter、优化器扩展、tools、可视化、FSDP | ✅ |
| **Phase 5 · 生态扩展** | 模型动物园、ONNX/TorchScript 导出、TTA、CI/CD | 🔜 |

## 相关资料

- [架构合同](../infrastructure/architecture.md)
- [训练指南](../guides/training.md)
- [扩展指南](../guides/extension.md)