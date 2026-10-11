# ElysiaCV

**统一计算机视觉训练框架**

ElysiaCV 是一个面向计算机视觉研究的中长期工程基础设施，不是单一模型代码库。
它为**分类、检测、分割**任务提供统一的配置系统、数据处理流水线、训练引擎和模型插件协议。

---

## 设计原则

| 原则 | 说明 |
|------|------|
| **多模型复用** | 所有模型以插件形式共存，通过自动发现注册，不修改核心代码 |
| **注册式零件** | Backbone / Neck / Head / Loss / Metric 通过统一 Registry 组装 |
| **统一训练流程** | 一套 TrainerLoop 驱动所有任务（分类/检测/分割），差异化只在 Runtime |
| **配置即代码** | YAML + `_base_` 继承 + CLI override，所有参数可追溯 |
| **可扩展性** | 新增模型只需创建 `plugin.py` + `runtime.py`，零框架改动 |

---

## 项目结构

```
ElysiaCV/
├── README.md
├── pyproject.toml                      # Python 项目定义
│
├── tools/                              # 统一入口脚本
│   ├── train.py                        # python tools/train.py config.yaml
│   ├── test.py                         # python tools/test.py config.yaml resume.pt
│   └── infer.py                        # python tools/infer.py config.yaml -i img.jpg
│
├── configs/                            # 用户配置（YAML + _base_ 继承）
│   ├── _base_/default_runtime.yaml     # DDP / DataLoader / precision 默认值
│   ├── classification/resnet50_imagenet.yaml
│   ├── detection/faster_rcnn_resnet50_coco.yaml
│   └── segmentation/fcn_resnet50_ade20k.yaml
│
├── docs/                               # 完整文档体系
│   ├── README.md                       # 文档入口
│   ├── status/                         # 当前状态与路线图
│   ├── guides/                         # 训练 / 评测 / 配置 / 扩展指南
│   └── infrastructure/                 # 架构合同
│
├── src/elysia_cv/                      # 核心 Python 包
│   ├── core/                           # Plugin 协议、Registry、异常体系
│   ├── config/                         # 配置系统（loader / override / validation）
│   ├── data/                           # 数据流水线（Dataset / Dataloader / Transforms）
│   ├── models/                         # 模型插件（classification / detection / segmentation）
│   ├── components/                     # 共享零件（backbones / necks / heads）
│   ├── losses/                         # 注册式 Loss 函数
│   ├── metrics/                        # 注册式 Metric
│   ├── training/                       # 训练引擎（Engine / Loop / Hooks / Distributed）
│   ├── optim/                          # 优化器 / 调度器
│   └── utils/                          # 通用工具
│
└── tests/                              # 80 个单元 + 端到端测试
```

---

## 快速开始

### 环境要求

```bash
conda create -n CV python=3.10 -y
conda activate CV
pip install torch torchvision
pip install -e .           # ElysiaCV editable install
```

### 训练

```bash
# 分类
python tools/train.py configs/classification/resnet50_imagenet.yaml

# 检测
python tools/train.py configs/detection/faster_rcnn_resnet50_coco.yaml \
    --set data.train.dataset_root=/path/to/coco optim.params.lr=0.01

# 分割
python tools/train.py configs/segmentation/fcn_resnet50_ade20k.yaml \
    --set data.train.dataset_root=/path/to/ade20k
```

### 测试

```bash
python tools/test.py configs/classification/resnet50_imagenet.yaml \
    ./work_dirs/checkpoint_final.pt
```

### 推理

```bash
python tools/infer.py configs/classification/resnet50_imagenet.yaml \
    checkpoints/best_Accuracy.pt -i img.jpg -o result.json
```

---

## 核心概念

### Model Plugin 三层架构

```
┌─────────────────────────────────────────────┐
│  plugin.py                                  │
│  MODEL_PLUGIN = SimpleModelPlugin(...)       │  ← 元数据（姓名/任务/规格）
│  自动发现，无需注册中心                       │
├─────────────────────────────────────────────┤
│  runtime.py                                 │
│  build_model(cfg) → nn.Module               │  ← 构建 + 训练/测试循环
│  forward_train(model, batch) → loss_dict    │
│  forward_test(model, batch) → pred_dict     │
├─────────────────────────────────────────────┤
│  自由目录                                    │
│  modeling.py / backbone.py / ...            │  ← 内部实现，无框架约束
└─────────────────────────────────────────────┘
```

### Registry 体系

| Registry | 用途 | 已注册 |
|----------|------|--------|
| `BACKBONES` | 视觉骨干网络 | ResNet |
| `NECKS` | 特征融合层 | FPN, IdentityNeck |
| `HEADS` | 任务输出头 | LinearClsHead, ClsHead, DetHead, SegHead |
| `LOSSES` | 损失函数 | CrossEntropy, Focal, SmoothL1, IoU, GIoU, Dice, Combined |
| `METRICS` | 评测指标 | Accuracy, Top5, MeanIoU, MeanAccuracy, MeanAP |
| `TRANSFORMS` | 数据增强 | Resize, RandomFlip, Normalize 等 13 种 |
| `HOOKS` | 训练回调 | Checkpoint, Logger, LrScheduler, Eval |

### 训练引擎架构

```
TrainingEngine
  ├── Config → TrainingPlan        (YAML + _base_ + override → 不可变计划)
  ├── ModelPlugin.discover()       (自动发现所有模型插件)
  ├── DataLoader + Transforms      (注册式 Pipeline)
  └── TrainerLoop
         ├── ExecutionStrategy      (梯度累积 + DDP no_sync)
         ├── HookBus                (Checkpoint / Logger / Eval / Scheduler)
         └── MetricsRuntime         (设备端累积 + 边界 all-reduce)
```

---

## 从 ElysiaRobot 继承的验证经验

ElysiaCV 的训练引擎基于 [ElysiaRobot](https://github.com/ElysiaRobot) 的 `native_engine.py` 简化重构，继承了以下 12 条已验证的性能优化：

- **DDP**: `bucket_cap_mb=64`, `gradient_as_bucket_view=true`, `broadcast_buffers=false`
- **梯度累积**: 仅最后 micro-step 同步梯度 + optimizer step
- **指标优化**: 设备端累积，log 边界批量 all-reduce（避免 400+ 次通信）
- **DataLoader**: `spawn` 线程防止 CUDA fork 死锁, `prefetch_factor=2`
- **CUDA**: `cudnn.benchmark=true`, `allow_tf32=true`, `zero_grad(set_to_none=true)`

详见 `docs/infrastructure/performance.md`。

---

## 扩展模型

```bash
# 创建新模型目录
mkdir -p src/elysia_cv/models/detection/retinanet

# plugin.py — 声明身份
cat > src/elysia_cv/models/detection/retinanet/plugin.py << 'EOF'
from elysia_cv.core.plugin import SimpleModelPlugin
from elysia_cv.models.contract import ModelSpec

MODEL_PLUGIN = SimpleModelPlugin(
    name="retinanet",
    task_type="detection",
    model_specs=[ModelSpec(name="retinanet_r50", ...)],
)
EOF

# runtime.py — 实现 build / train / test
cat > src/elysia_cv/models/detection/retinanet/runtime.py << 'EOF'
def build_model(cfg): ...
def forward_train(model, batch, *, device=None): ...
def forward_test(model, batch, *, device=None): ...
EOF
```

新增后无需修改任何框架代码，`MODEL_PLUGIN` 会被自动发现。

---

## 测试

```bash
pytest tests/ -v    # 80 个测试, ~30s
```

---

## License

Apache-2.0 — 详见 `LICENSE` 文件。