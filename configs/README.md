# 配置索引

> 状态：Active（随配置变更更新） | 最后更新：2026-10-20

---

## 配置目录

```
configs/
├── README.md
├── _base_/
│   └── default_runtime.yaml          # 所有配置的基础模板
├── models/                            # 模型规格定义
│   ├── mlp.yaml
│   ├── simple_cnn.yaml
│   ├── resnet18.yaml .. resnet101.yaml
│   ├── vit_b16.yaml .. vit_l16.yaml
│   ├── swin_tiny.yaml .. swin_base.yaml
│   └── fcos_r50.yaml
├── datasets/                          # 数据集规格定义
│   └── cassava.yaml
├── experiments/                       # 模型 + 数据集组合实验
│   ├── cassava_mlp.yaml .. cassava_vit_b16.yaml
│   └── herbs_swin_tiny.yaml
├── classification/                    # 独立任务配置
│   └── resnet50_imagenet.yaml
├── detection/
│   └── fcos_resnet50_coco.yaml       # FCOS COCO 检测
└── segmentation/
    └── fcn_resnet50_ade20k.yaml      # FCN ADE20K 分割
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
| `engine.optimization` | gradient_accumulation_steps=1 |
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

### 单文件

在对应任务目录下创建 `.yaml`，所有配置写在一个文件中（适合快速原型）。参考 `classification/resnet50_imagenet.yaml`。