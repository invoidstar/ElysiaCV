# 当前状态

> 状态：Active（随工程进展更新） | 最后更新：2026-10

---

## 版本

ElysiaCV **v0.1.0** — 首个工程验证版本。

## 功能清单

### Core & Config

| 功能 | 状态 | 说明 |
|------|------|------|
| Registry 系统 | ✅ | 10 个全局 Registry (backbone/neck/head/transform/loss/metric/hook/optimizer/scheduler) |
| Plugin 自动发现 | ✅ | 扫描 `models/*/plugin.py`，支持 flat + task-grouped 两种布局 |
| `_base_` 配置继承 | ✅ | 递归 YAML 合并，子配置覆盖父配置 |
| CLI override | ✅ | `--set key=value` 支持 null/bool/int/float/json/list |
| 环境变量展开 | ✅ | `${VAR}` 和 `${VAR:-default}` |
| Fail-fast 校验 | ✅ | 未知 key 立即拒绝 + 拼写建议 |

### 数据流水线

| 功能 | 状态 | 说明 |
|------|------|------|
| BaseDataset | ✅ | 通用 Map-style Dataset |
| ImageFolderDataset | ✅ | 自动扫描 class 子目录 |
| DataLoader | ✅ | 继承 ElysiaRobot 的 DDP spawn 修复 + 默认参数 |
| Transforms (13 种) | ✅ | LoadImage/Resize/Flip/Normalize/Pack 等，全部注册式 |
| DistributedSampler | ✅ | 确定性 shuffle per epoch |

### 模型

| 模型 | 任务 | Backbone | Neck | 状态 |
|------|------|----------|------|------|
| resnet (18/34/50/101) | classification | ResNet | — | ✅ |
| faster_rcnn (r50/r101) | detection | ResNet | FPN | ✅ |
| fcn (r50/r101) | segmentation | ResNet | IdentityNeck | ✅ |

### 组件零件

| 类型 | 组件 | 状态 |
|------|------|------|
| Backbone | ResNet (torchvision, 5 种深度) | ✅ |
| Neck | FPN, IdentityNeck | ✅ |
| Head | LinearClsHead, ClsHead, DetHead, SegHead | ✅ |

### 损失函数

| Loss | 适用任务 | 状态 |
|------|---------|------|
| CrossEntropyLoss | 分类/分割 | ✅ |
| FocalLoss | 分类 (class imbalance) | ✅ |
| SmoothL1Loss | 检测 (bbox regression) | ✅ |
| IoULoss / GIoULoss | 检测 | ✅ |
| DiceLoss | 分割 | ✅ |
| CombinedSegLoss (CE + Dice) | 分割 | ✅ |

### 评测指标

| Metric | 适用任务 | 状态 |
|--------|---------|------|
| Accuracy / Top5Accuracy | 分类 | ✅ |
| MeanIoU / MeanAccuracy | 分割 | ✅ |
| MeanAP (stub) | 检测 | ✅ |

### 训练引擎

| 功能 | 状态 | 说明 |
|------|------|------|
| TrainerLoop | ✅ | 设备无关训练循环 |
| ExecutionStrategy | ✅ | 梯度累积 + DDP no_sync 决策 |
| Distributed (DDP) | ✅ | init / wrap / all-reduce / cudnn 配置 |
| CheckpointHook | ✅ | 定期保存 + best metric + max_keep 清理 |
| LoggerHook | ✅ | loss / LR / step time 日志 |
| LrSchedulerHook | ✅ | 每步 step scheduler |
| EvalHook | ✅ | 定期验证 + best metric 联动 |
| MetricsRuntime | ✅ | 设备端累积 + 边界 bulk all-reduce |
| GPU Monitor | ✅ | nvidia-smi 解析 + 窗口统计 |
| Optimizer/Scheduler | ✅ | build_optimizer + build_scheduler via registry |
| TrainingPlan | ✅ | CVConfig → 不可变 resolved plan 编译器 |

---

## 路线图

### P0-P5: ✅ 全部完成 (80 tests)

### P6: 🔜 社区模型接入（暂缓）

| 任务 | 优先级 |
|------|--------|
| YOLO 检测模型接入 | 低 |
| DETR 检测模型接入 | 低 |
| SAM 分割模型接入 | 低 |

### 远期

| 方向 | 说明 |
|------|------|
| torch.compile 长训验证 | 已预留选项，待 20k+ step 验证 |
| FSDP 支持 | 大模型训练策略 |
| ONNX/TensorRT 导出 | 推理优化 |