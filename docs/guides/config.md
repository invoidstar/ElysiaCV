# 配置指南

> 状态：Active（长期有效） | 最后更新：2026-10

---

## 快速开始

一个最小可训练的配置需要以下节：

```yaml
_base_: "../_base_/default_runtime.yaml"

run:
  name: my_experiment
  stage: train
  output_dir: ./outputs/checkpoints/my_experiment

model:
  type: resnet50
  backbone:
    type: ResNet
    depth: 50
  head:
    type: LinearClsHead
    num_classes: 1000

data:
  train:
    dataset_root: /data/mydataset/train
    batch_size: 32
    transforms:
      - type: Resize
        size: 224
      - type: Normalize
      - type: PackInputs
  val:
    dataset_root: /data/mydataset/val
    batch_size: 32
    transforms:
      - type: Resize
        size: 224
      - type: Normalize
      - type: PackInputs

optim:
  type: SGD
  params:
    lr: 0.01
    momentum: 0.9

schedule:
  type: CosineAnnealing
  steps: 100000
  base_lr: 0.01

eval:
  eval_freq: 5000
  metrics:
    - Accuracy
```

---

## `_base_` 继承

`_base_` 引用会被递归加载并 deep-merge，子配置覆盖父配置。

```yaml
# _base_/default_runtime.yaml
engine:
  precision:
    mixed_precision: true
```

```yaml
# my_config.yaml
_base_: "../_base_/default_runtime.yaml"
engine:
  precision:
    mixed_precision: false    # 覆盖 base
```

**继承链示例**：

```
my_config.yaml
  → _base_/default_runtime.yaml
```

---

## CLI Override

所有配置值都可以通过 `--set` 在命令行覆盖：

```bash
python tools/train.py config.yaml --set key=value
```

### 值类型

| 示例 | 解析结果 |
|------|---------|
| `--set a=true` | `True` |
| `--set a=false` | `False` |
| `--set a=42` | `42` |
| `--set a=3.14` | `3.14` |
| `--set a=null` | `None` |
| `--set a=\"hello\"` | `"hello"` |
| `--set a='[1,2,3]'` | `[1, 2, 3]` |

### 嵌套路径

```bash
# 覆盖嵌套值
--set data.train.batch_size=64
--set optim.params.lr=0.001
--set schedule.warmup_steps=1000
```

### 未知 key 检测

拼写错误会被立即拒绝：

```bash
$ python tools/train.py config.yaml --set schedule.wampup_steps=1000
Error: unknown key 'schedule.wampup_steps'. Did you mean 'schedule.warmup_steps'?
```

---

## 环境变量

配置文件中可以使用 `${VAR}` 和 `${VAR:-default}` 语法：

```yaml
data:
  train:
    dataset_root: ${IMAGENET_ROOT:-/data/imagenet}
    # 从环境变量读取，不存在时用 /data/imagenet
```

---

## 配置节参考

### `run` — 运行元数据

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 实验名称 |
| `stage` | `str` | `train` / `test` |
| `seed` | `int` | 随机种子（默认 42） |
| `output_dir` | `str` | checkpoint + 日志输出目录 |
| `test_run` | `bool` | 仅运行 10 步（快速验证） |

### `model` — 模型定义

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `str` | 模型插件名称 (如 `resnet50`) |
| `pretrained_path` | `str` | 预训练权重路径 |
| `backbone` | `dict` / `str` | Backbone 配置 (分类/检测用 dict；动作识别用 string 如 "resnet50") |
| `neck` | `dict` | Neck 配置 (检测/分割需要) |
| `head` | `dict` | Head 配置 (type + 参数) |
| `variant` | `str` | 模型变体 (YOLO/X3D/SlowFast 等需要) |
| `num_temp_layers` | `int` | temporal transformer 层数 (X-CLIP) |
| `adapter_ratio` | `int` | adapter 压缩比 (ST-Adapter) |
| `freeze_backbone` | `bool` | 冻结 backbone 参数 (X-CLIP/ST-Adapter) |

### `data` — 数据

| 字段 | 类型 | 说明 |
|------|------|------|
| `train.dataset_root` | `str` | 训练集路径 |
| `train.batch_size` | `int` | 训练 batch size |
| `train.num_workers` | `int` | DataLoader worker 数量 |
| `train.transforms` | `list` | Transform pipeline（分类/检测用图像 transforms；动作识别用 DecordInit/SampleFrames/VideoResize 等视频 transforms） |
| `train.format` | `str` | 视频数据集格式：`ucf101` / `kinetics` |
| `val.*` | — | 验证集同理 |

### `engine` — 训练策略

| 字段 | 类型 | 说明 |
|------|------|------|
| `optimization.gradient_accumulation_steps` | `int` | 梯度累积步数 |
| `optimization.grad_clip_norm` | `float` | 梯度裁剪阈值 |
| `precision.mixed_precision` | `bool` | 开启混合精度 |
| `precision.dtype` | `str` | `float32` / `bfloat16` |
| `distributed.ddp_find_unused_parameters` | `bool` | DDP unused param 检测 |
| `compile.enabled` | `bool` | 开启 torch.compile |

### `optim` — 优化器

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `str` | `SGD` / `AdamW` |
| `params` | `dict` | 传递给优化器的参数 (lr, momentum, weight_decay, ...) |

### `schedule` — 学习率策略

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `str` | `CosineAnnealing` / `MultiStepLR` |
| `steps` | `int` | 总训练步数 |
| `base_lr` | `float` | 基础学习率 |
| `min_lr` | `float` | 最小学习率 |
| `warmup_steps` | `int` | warmup 步数 |
| `params` | `dict` | scheduler 额外参数 |

### `eval` — 评测

| 字段 | 类型 | 说明 |
|------|------|------|
| `eval_freq` | `int` | 每 N 步评测一次 |
| `metrics` | `list[str]` | 评测指标 |
| `save_best` | `str` | 保存最佳模型的指标名 |
| `save_final` | `bool` | 训练结束时保存 checkpoint |