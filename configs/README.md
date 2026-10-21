# 配置索引

> 状态：Active（随配置变更更新） | 最后更新：2026-10-21

---

## 配置目录

```
configs/
├── README.md
├── _base_/
│   └── default_runtime.yaml          # 所有配置的基础模板（分类/检测）
├── models/                            # 模型规格定义
│   ├── mlp.yaml
│   ├── resnet18.yaml .. resnet50.yaml
│   ├── vit_b16.yaml .. vit_l16.yaml
│   ├── swin_tiny.yaml .. swin_base.yaml
├── datasets/                          # 数据集规格定义
│   ├── cassava.yaml
│   ├── ade20k.yaml
│   └── cityscapes.yaml
├── experiments/                       # 模型 + 数据集组合实验
│   ├── cassava_*.yaml
│   ├── ade20k_fcn_r50.yaml
│   └── cityscapes_fcn_r50.yaml
├── classification/                    # 独立分类任务配置
│   └── resnet50_imagenet.yaml
├── detection/                         # 检测任务配置
│   ├── fcos_resnet50_coco.yaml
│   └── _base_/                        # YOLO 基础配置（22 个变体）
├── segmentation/                      # 分割任务配置
│   └── fcn_resnet50_ade20k.yaml
└── action_recognition/                # 动作识别配置（11 种模型）
    ├── _base_/video_base.yaml
    ├── tsn_resnet50_ucf101.yaml
    ├── tsm_resnet50_ucf101.yaml
    ├── mvitv2_small_ucf101.yaml
    ├── r3d18_ucf101.yaml
    ├── x3d_xs_ucf101.yaml
    ├── xclip_vitb16_ucf101.yaml
    ├── st_adapter_vitb16_ucf101.yaml
    └── uniformerv2_ucf101.yaml
```

配置采用**模型+数据集分离组合**模式：`experiments/cassava_resnet50.yaml` 通过 `_base_` 组合 `models/resnet50.yaml` + `datasets/cassava.yaml`，减少重复配置。

---

## `_base_/default_runtime.yaml`

所有配置通过 `_base_: "../_base_/default_runtime.yaml"` 继承的公共默认值：

| 节 | 内容 |
|----|------|
| `engine.distributed` | DDP 默认参数（bucket_cap_mb=64, gradient_as_bucket_view=true） |
| `engine.precision` | CUDA 后端配置（cudnn.benchmark=true, allow_tf32=true） |
| `engine.compile` | torch.compile 默认关闭 |
| `engine.optimization` | gradient_accumulation_steps=1, ema_enable=false, ema_decay=0.9999 |
| `engine.strategy` | ddp / fsdp / single |
| `tracking` | log_freq=50, wandb 关闭 |
| `run` | seed=42, test_run=false |
| `eval` | eval_freq=5000, save_final=true |

---

## 使用方式

```bash
# 组合式配置（推荐 — 模型+数据集分离）
python tools/train.py configs/experiments/cassava_resnet50.yaml

# 独立任务配置
python tools/train.py configs/classification/resnet50_imagenet.yaml

# 断点续训
python tools/train.py configs/experiments/cassava_resnet50.yaml --resume

# 覆盖参数
python tools/train.py configs/detection/fcos_resnet50_coco.yaml \
    --set data.train.dataset_root=/data/coco schedule.steps=50000

# 查看完整展开配置
elysia config dump configs/experiments/cassava_resnet50.yaml

# 多卡 (torchrun)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    tools/train.py configs/experiments/cassava_resnet50.yaml
```

---

## 新增配置

### 组合式（推荐）

1. **模型配置** `configs/models/my_model.yaml`：
```yaml
model:
  type: resnet50
  backbone: {type: ResNet, depth: 50}
  head: {type: LinearClsHead, num_classes: 5}
```

2. **数据集配置** `configs/datasets/my_data.yaml`：
```yaml
data:
  train: {dataset_root: /path/to/data, batch_size: 32, ...}
```

3. **实验配置** `configs/experiments/my_experiment.yaml`：
```yaml
_base_: [../models/my_model.yaml, ../datasets/my_data.yaml]

run:
  name: my_experiment
  output_dir: ./outputs/checkpoints/my_experiment

optim: {type: AdamW, params: {lr: 0.001}}
schedule: {type: CosineAnnealing, steps: 10000}
eval: {eval_freq: 1000, metrics: [{type: Accuracy}]}
```

### 启用训练技巧

```yaml
# Model EMA（引擎自动启用）
engine:
  optimization:
    ema_enable: true
    ema_decay: 0.9999

# 数据增强（per-sample + batch 级）
data:
  train:
    transforms:
      - {type: LoadImageFromFile}
      - {type: Resize, size: 224}
      - {type: RandomRotation, degrees: 10, prob: 0.5}
      - {type: RandomFlip, prob: 0.5}
      - {type: Normalize}
      - {type: PackInputs}
    batch_transforms:
      - {type: MixUp, alpha: 0.2, prob: 0.5}
      - {type: CutMix, alpha: 1.0, prob: 0.5}
```

### 单文件

在对应任务目录下创建 `.yaml`，所有配置写在一个文件中（适合快速原型）。参考 `classification/resnet50_imagenet.yaml`。