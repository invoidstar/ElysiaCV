# AGENTS.md — ElysiaCV

> AI agent 和贡献者：在修改代码之前，请先阅读本文档。
> 最后更新：2026-10-20

---

## 1. 项目概览

ElysiaCV 是一个面向计算机视觉（分类、检测、分割）的统一训练框架。
核心理念：**"让视觉研究聚焦模型，让工程基础设施持续复用"**。

**原则**：
- **配置驱动**：一切从 YAML 配置出发（MMDetection 式的 `_base_` 继承）
- **插件式架构**：模型通过装饰器自注册，无需集中注册文件
- **组件库**：共享的 Backbone、Neck、Head、Loss、Metric
- **训练引擎**：TrainerLoop + HookBus（checkpoint/log/evaluation/EMA/training-curves），支持 DDP / FSDP

**版本**：v0.2.0 Alpha | **Python**：≥3.10 | **PyTorch**：≥2.0

---

## 2. 项目结构

> **⼯程侧** (framework) 与 **项目侧** (project) 在代码上同居一个仓库，靠文档区分。
> 类似于 ElysiaRobot（框架）与 mstvla4resta（使用框架的项目）的关系。

```
ElysiaCV/
│
│  ╔══════════════════════════════════════════════════════════════╗
│  ║  🏗️ 工程侧 — 可复用框架代码 (src/elysia_cv/)                ║
│  ╠══════════════════════════════════════════════════════════════╣
│  ║                                                              ║
├──╢ src/elysia_cv/                                               ║
│  ║ ├── core/           零依赖基础设施 (Registry, Plugin, Errors) ║
│  ║ ├── config/         配置加载、_base_ 合并、校验               ║
│  ║ ├── components/     共享 Backbone / Neck / Head 组件库        ║
│  ║ ├── losses/         损失函数 (FocalLoss, CrossEntropy...)     ║
│  ║ ├── metrics/        评估指标 (MeanAP, MeanIoU, Accuracy)       ║
│  ║ ├── optim/          优化器与调度器构建器 (SGD/Adam/AdamW/RMSprop)║
│  ║ ├── data/           通用数据加载层 (COCO/ImageFolder/Adapter) ║
│  ║ │   └── transforms/  per-sample + batch 级变换 (MixUp/CutMix/Mosaic)║
│  ║ ├── models/         模型契约 + 参考插件 (resnet/fcos/fcn)     ║
│  ║ ├── training/       训练引擎 (Loop, Hooks, DDP/FSDP, EMA, Engine)║
│  ║ ├── utils/          可视化 (PR curves, bbox, segmentation)    ║
│  ║ └── cli.py          统一 CLI (train/test/infer/config dump)   ║
│  ║                                                              ║
│  ╚══════════════════════════════════════════════════════════════╝
│
├── .github/workflows/ci.yml  ← CI: lint + test + coverage
├── tools/train.py / test.py / infer.py  ← 训练/评测/推理入口
│
│  ╔══════════════════════════════════════════════════════════════╗
│  ║  📦 项目侧 — Cassava 木薯叶片病害分类实验                     ║
│  ╠══════════════════════════════════════════════════════════════╣
│  ║  详见 projects/HERBS/AGENTS.md                              ║
│  ║                                                              ║
├──╢ configs/                                                     ║
│  ║ ├── models/mlp.yaml .. swin_base.yaml   模型定义 (9 个)       ║
│  ║ ├── datasets/cassava.yaml              Cassava 数据集         ║
│  ║ └── experiments/cassava_*.yaml         实验组合 (9 个)        ║
│  ║                                                              ║
├──╢ scripts/run_cassava_experiments.sh     批量实验启动脚本        ║
├──╢ tools/analyze.py                       训练后分析              ║
├──╢ projects/HERBS/                        项目文档目录            ║
│  ║ ├── AGENTS.md                          项目约定与实验记录      ║
│  ║ └── tools/analyze.py                   混淆矩阵 + GradCAM     ║
│  ║                                                              ║
│  ╚══════════════════════════════════════════════════════════════╝
│
├── tests/                         pytest (10 套, 163 tests)
├── docs/                          指南和架构文档
└── AGENTS.md                      ← 你在这里（根文档）
```

---

## 3. 核心架构原则

### 3.1 模块依赖方向

```
core/  ←  config/  ←  data/  ←  models/  ←  training/
  ↓         ↓          ↓         ↓            ↓
  [所有模块都可以 import core/ 和 config/]

components/  ←  由 models/ 通过 Registry 引用（不直接 import）
losses/      ←  由 models/*/runtime.py 使用
metrics/     ←  由 training/ 使用
optim/       ←  由 cli.py / training/ 使用
```

**规则**：
- `core/` 不 import torch 代码，不 import 项目内代码
- `config/` 只 import `core/`
- `components/` 通过 `@BACKBONES.register_module()` 注册，绝不 import `models/`
- `models/*/runtime.py` 负责**前向计算逻辑**
- `training/loop.py` 负责**训练循环**（step、backward、optimizer）
- `cli.py` 是**唯一入口**，负责串联 model + data + training

### 3.2 Registry 模式

所有组件通过装饰器自发现，绝不通过 import 集中列出：

```python
from elysia_cv.core.registry import BACKBONES, HEADS, LOSSES

@BACKBONES.register_module()
class ResNet(nn.Module):
    ...

# 构建时
backbone = BACKBONES.build({"type": "ResNet", "depth": 50})
```

**全局 Registry 列表**：`BACKBONES`、`NECKS`、`HEADS`、`LOSSES`、`METRICS`、`TRANSFORMS`、`HOOKS`

`TRANSFORMS` 支持 per-sample（RandomFlip、ColorJitter、RandomRotation、RandomAffine）和 batch 级（MixUp、CutMix、Mosaic）两种变换。

### 3.3 配置组合模型

```
models/xxx.yaml          datasets/xxx.yaml
       \                      /
        \     _base_          /
         └───→ experiment.yaml ←─── _base_/default_runtime.yaml
```

- `_base_` 支持列表：`_base_: [../a.yaml, ../b.yaml]`，从左到右合并，子节点覆盖父节点
- `_deep_merge` 递归合并嵌套 dict；列表按索引合并
- 环境变量：字符串中支持 `${VAR}` 和 `${VAR:-default}`

---

## 4. 常用操作

### 4.1 环境准备

```bash
conda activate CV
pip install -e ".[dev]"
```

### 4.2 运行测试

```bash
pytest tests/ -v                          # 全部 163 tests
pytest tests/test_cassava_experiments.py -v  # 单个测试
CUDA_VISIBLE_DEVICES=6 pytest tests/ -v   # 指定 GPU
```

### 4.3 代码检查

```bash
ruff check src/ tests/ tools/
black --check src/ tests/ tools/
```

配置：120 字符行宽、py310 目标、规则 `A,B,E,F,I,W`。

### 4.4 CI/CD

`.github/workflows/ci.yml` 每次 push/PR 到 main 自动运行：
- **Lint**: `ruff check src/ tests/ tools/`
- **Test**: `pytest tests/` on Python 3.10 + 3.11 (CPU)
- **Coverage**: `pytest --cov --cov-fail-under=30` + HTML artifact

### 4.5 运行实验

```bash
# 单卡
python tools/train.py configs/experiments/cassava_resnet50.yaml

# 单卡 + 断点续训
python tools/train.py configs/experiments/cassava_resnet50.yaml --resume

# 多卡 (torchrun)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    tools/train.py configs/experiments/cassava_resnet50.yaml

# 带覆盖
python tools/train.py configs/experiments/cassava_vit_b16.yaml \
    --set schedule.steps=100 optim.params.lr=0.001
```

### 4.6 评测

```bash
python tools/test.py configs/experiments/cassava_resnet50.yaml \
    --checkpoint outputs/checkpoints/xxx/checkpoint_best.pt

python tools/infer.py configs/experiments/cassava_resnet50.yaml \
    -c outputs/checkpoints/xxx/checkpoint_best.pt -i image.jpg

elysia config dump configs/experiments/cassava_resnet50.yaml

bash projects/HERBS/scripts/run.sh
```

### 4.7 训练后分析

```bash
python tools/analyze.py configs/experiments/cassava_resnet50.yaml \
    --checkpoint outputs/checkpoints/cassava_resnet50/checkpoint_best.pt
```

输出：`analysis/confusion_matrix.png`、`analysis/per_class_report.txt`、`analysis/gradcam/`、`analysis/training_curves.png`（自动生成）、`analysis/pr_curves.png`（需手动调用 `plot_pr_curves()`）

---

## 5. 添加新模块

### 5.1 添加新模型

1. **创建模型配置**：`configs/models/my_model.yaml`
2. **创建运行时**：`src/elysia_cv/models/classification/my_model/plugin.py` + `runtime.py`
3. **创建实验配置**：`configs/experiments/my_experiment.yaml`
4. **注册组件**（如新 backbone）：在 `src/elysia_cv/__init__.py` 中添加 import
5. **添加测试**：`tests/test_my_model.py`

### 5.2 添加新数据集

1. **创建数据集配置**：`configs/datasets/my_data.yaml`
2. **创建实验**：`_base_: [../models/X.yaml, ../datasets/my_data.yaml]`

---

## 6. 代码约定

| 约定 | 说明 |
|------|------|
| 类型标注 | 每个 `.py` 顶部加 `from __future__ import annotations` |
| 文档字符串 | Google 风格，每个公开类/函数都要有 |
| 命名 | `snake_case` 函数/变量，`PascalCase` 类，`UPPER_CASE` 常量 |
| 导入顺序 | stdlib → 第三方 → 内部，每组间空一行 |
| GPU | 始终用 `device` 参数，不硬编码 `"cuda:0"`；数据传输用 `non_blocking=True` |
| 打印 | 管道输出场景用 `flush=True` |
| 错误处理 | raise `elysia_cv.core.errors` 中的异常，不裸抛 `Exception` |

---

## 7. Git 提交约定

```
feat:     新功能
fix:      Bug 修复
refactor: 代码重构（行为不变）
docs:     文档
test:     测试
chore:    构建、CI、依赖
style:    格式化、lint（不改变代码逻辑）
```

---

## 8. 已解决的问题

| 问题 | 状态 | 详情 |
|------|------|------|
| CLI ↔ TrainerLoop 统一 | ✅ | `cli.py` 已使用 `TrainingEngine` → `TrainerLoop` + `HookBus` |
| 配置模型/数据集分离 | ✅ | 拆分为 `configs/{models,datasets,experiments}/` |
| DDP 支持 | ✅ | `TrainingEngine` 内置 DDP init + `wrap_ddp` + `no_sync` |
| FSDP 支持 | ✅ | `wrap_fsdp` + `fsdp_state_dict`，engine 中 strategy 自动路由 |
| `test.py`/`infer.py` | ✅ | 真实实现：checkpoint 加载 + eval 循环 + 分类/检测路由 |
| Detection runtime | ✅ | FCOS anchor-free：center-sampling 真损失 + NMS 后处理 |
| MeanAP metric | ✅ | COCO-standard 101-pt mAP / mAP_50 / mAP_75 |
| `--resume` 断点续训 | ✅ | 自动恢复 model/optimizer/scheduler/step |
| MessageHub | ✅ | 全局标量状态总线 + best-value 追踪 |
| DataPrefetcher | ✅ | CUDA stream 异步预取批次到 GPU |
| config dump | ✅ | `elysia config dump` 展开 _base_ 继承 |
| Data Adapter | ✅ | HF / torchvision 分类+检测格式适配器 |
| 检测可视化 | ✅ | `draw_bboxes` + `draw_segmentation_overlay` |
| 数据增强 | ✅ | MixUp/CutMix/Mosaic + RandomRotation/RandomAffine + batch 级管线 |
| Model EMA | ✅ | 影子权重 + ramp-up decay + apply_shadow/restore，eval 自动切换 |
| PR 曲线 & 训练曲线 | ✅ | `plot_pr_curves()` / `MetricHistory` Hook → PNG |
| CI/CD | ✅ | GitHub Actions: lint + test + coverage (≥30%) |

详见 `docs/status/current.md`。