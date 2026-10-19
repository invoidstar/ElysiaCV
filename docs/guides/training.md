# 训练指南

> 状态：Active（长期有效） | 最后更新：2026-10

---

## 入口命令

```bash
python tools/train.py <config.yaml> [--set key=value ...] [--resume checkpoint.pt]
```

---

## 基本用法

### 启动训练

```bash
# 分类（组合配置：模型 + 数据集）
python tools/train.py configs/experiments/cassava_resnet50.yaml

# 或直接指定模型和数据集（旧格式）
python tools/train.py configs/classification/resnet50_imagenet.yaml

# 检测
python tools/train.py configs/detection/fcos_resnet50_coco.yaml

# 分割
python tools/train.py configs/segmentation/fcn_resnet50_ade20k.yaml
```

### 覆盖配置

```bash
python tools/train.py configs/classification/resnet50_imagenet.yaml \
    --set data.train.dataset_root=/data/imagenet \
    --set schedule.base_lr=0.1 \
    --set optim.params.weight_decay=0.0001
```

多参数覆盖会递归合并到已加载的配置中。

### 从 Checkpoint 恢复

```bash
python tools/train.py configs/classification/resnet50_imagenet.yaml \
    --resume ./outputs/checkpoints/checkpoint_step_10000.pt
```

恢复时模型权重、优化器状态、scheduler 状态全部一致。

---

## 核心概念

### 训练步骤

| 概念 | 说明 |
|------|------|
| `micro_step` | 一次前向 + 反向（不计入 global_step） |
| `global_step` | 一次 optimizer.step() 后 +1 |
| `epoch` | 遍历一轮数据集（data loop 中 epoch 边界） |
| `grad_accum_steps` | 多少次 micro_step 触发一次 optimizer step |

```
micro_step_0   micro_step_1   micro_step_2   micro_step_3
  (no_sync)      (no_sync)      (no_sync)      (sync + step)
                                               ↑ global_step ++
```

### 配置文件结构

```yaml
_base_: "../_base_/default_runtime.yaml"    # 继承默认值

run:        # 运行元数据
model:      # 模型定义 (type + backbone + neck + head + loss_weights)
data:       # 数据源 + 预处理
  train:
    dataset_root: /data/imagenet/train
    batch_size: 256
    transforms: [...]        # per-sample 变换（Normalize, Resize, RandomFlip...）
    batch_transforms: [...]  # batch 级变换（MixUp, CutMix, Mosaic...）
  val:
    dataset_root: /data/imagenet/val
    batch_size: 64
    transforms: [...]
engine:     # 训练策略 (DDP / precision / compile / optimization / EMA / SWA)
optim:      # 优化器 (type + params)
schedule:   # 学习率策略 (type + steps + warmup)
tracking:   # 日志 / W&B
eval:       # 评测频率 + 指标 + early_stop
```

---

## 训练技巧

### Model EMA（指数滑动平均）

在训练时维护一份模型权重的指数滑动平均，评测时自动切换到 EMA 权重，通常可提升 0.5-1% 精度。

```yaml
engine:
  optimization:
    ema_enable: true       # 启用
    ema_decay: 0.9999      # 衰减系数（可选，默认 0.9999）
```

### SWA（随机权重平均）

训练后期周期性保存模型权重 snapshot，训练结束时 merge 为平均权重。与 EMA 互补，可额外提 0.3-0.5%。

```yaml
engine:
  optimization:
    swa_enable: true
    swa_start_step: 9000   # 从第 N 步开始收集 snapshot
    swa_interval: 500      # 每 N 步收集一次
```

评测时自动切换到 SWA 平均权重（叠加在 EMA 之上）。

### EarlyStopping（自动终止）

当监控的评测指标连续 N 轮无提升时，自动终止训练。

```yaml
eval:
  early_stop_enable: true
  early_stop:
    metric: Accuracy       # 监控指标
    mode: max              # max（越大越好）/ min（越小越好）
    patience: 5            # 容忍 N 轮不提升
    min_delta: 0.001       # 最小提升阈值
```

### TTA（测试时增强）

推理时 multi-scale + horizontal flip 投票预测。分类任务自动支持：

```python
from elysia_cv.data.transforms.tta import tta_predict

logits = tta_predict(model, image, device,
                     scales=[0.8, 1.0, 1.2],    # 3 种缩放
                     flips=[False, True])         # × 水平翻转 = 6 种变体
pred_class = logits.argmax(dim=1).item()
```

### 类别不平衡处理

```python
from elysia_cv.data.sampler import ClassBalancedSampler, compute_class_weights

# 类平衡采样器
sampler = ClassBalancedSampler(dataset, num_samples=1000)
loader = DataLoader(dataset, sampler=sampler, batch_size=32)

# 类别加权 loss 权重
weights = compute_class_weights(dataset, beta=0.9999)  # effective-number
loss_fn = nn.CrossEntropyLoss(weight=weights)
```

### 数据缓存

小数据集可全量预加载到内存，加速 I/O：

```python
from elysia_cv.data.cache import CachedDataset

ds = CachedDataset(original_dataset)                  # 内存缓存
ds = CachedDataset(original_dataset,
                   cache_dir="/tmp/my_cache.lmdb",
                   cache_mode="lmdb")                 # 持久化缓存
```

### 分割数据集 (ADE20K / Cityscapes)

```yaml
data:
  train:
    type: ADE20KDataset     # 或 CityscapesDataset
    root: /data/ADEChallengeData2016
    split: training          # ADE20K: training/validation
    transforms:
      - {type: Resize, size: [512, 512]}
      - {type: Normalize}
      - {type: PackSegInputs}
```

### 数据增强

**Per-sample 变换**（对每张图独立应用）：

| 变换 | 说明 | 配置示例 |
|------|------|----------|
| RandomFlip | 随机水平/垂直翻转 | `{type: RandomFlip, prob: 0.5}` |
| ColorJitter | 亮度/对比度/饱和度/色调 | `{type: ColorJitter, brightness: 0.2}` |
| RandomRotation | 旋转 ±N 度 + bbox 同步 | `{type: RandomRotation, degrees: 10}` |
| RandomAffine | 平移/缩放/剪切 + bbox 同步 | `{type: RandomAffine, translate: [0.1,0.1], scale: [0.9,1.1]}` |
| RandomErasing | 随机擦除矩形区域 | `{type: RandomErasing, prob: 0.3}` |

**Batch 级变换**（对整个 batch 操作，混合同 batch 的样本）：

| 变换 | 说明 | 适用任务 | 配置示例 |
|------|------|----------|----------|
| MixUp | 两图线性插值，标签按比例混合 | 分类 | `{type: MixUp, alpha: 0.2, prob: 0.5}` |
| CutMix | 裁切一块贴到另一张图 | 分类 | `{type: CutMix, alpha: 1.0, prob: 0.5}` |
| Mosaic | 4 图拼 1 图，bbox 同步 | 检测 | `{type: Mosaic, output_size: [640, 640], prob: 0.5}` |

完整配置示例：

```yaml
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
      - {type: CutMix, alpha: 1.0, prob: 0.3}
```

### 训练曲线自动记录

`MetricHistory` Hook 自动记录每个 eval step 的指标值，训练结束时生成 `analysis/training_curves.png`。

无需配置 — 引擎自动启用。

---

## Checkpoint 策略

| 触发方式 | 命名格式 | 示例 |
|---------|---------|------|
| 间隔保存 | `checkpoint_step_{N}.pt` | `checkpoint_step_5000.pt` |
| 最终保存 | `checkpoint_final.pt` | 训练结束时 |
| 最佳指标 | `checkpoint_best_{metric}.pt` | `checkpoint_best_Accuracy.pt` |

默认保留最近 5 个文件。

---

## 日志

日志格式：

```
   Step      Loss        LR  Time/step    Elapsed
    100    2.3456  0.010000     0.234s      23.4s
    200    1.9876  0.009500     0.231s      46.6s
```

| 列 | 说明 |
|----|------|
| `Step` | `global_step` |
| `Loss` | 当前 step 的平均 loss |
| `LR` | 当前学习率 |
| `Time/step` | 自上次日志以来的平均每步时间 |
| `Elapsed` | 自训练开始的总时间 |

---

## 分布式训练

```bash
# 单机多卡
torchrun --nproc_per_node=8 tools/train.py configs/.../xxx.yaml

# 多机多卡
torchrun --nnodes=2 --nproc_per_node=8 --master_addr=... tools/train.py ...
```

DDP 配置在 `configs/_base_/default_runtime.yaml` 中已预设最优值：

```yaml
engine:
  distributed:
    ddp_find_unused_parameters: true
    ddp_broadcast_buffers: false
    ddp_bucket_cap_mb: 64
    ddp_gradient_as_bucket_view: true
```

---

## 常见问题

### DataLoader 卡死

如果 DDP 训练卡在 DataLoader 阶段不启动：

使用 `--set engine.optimization.gradient_accumulation_steps=1` 先排除梯度累积问题，然后检查 `num_workers` 是否为 0。

### GPU 内存不足

```bash
# 方法 1: 梯度累积
--set engine.optimization.gradient_accumulation_steps=4

# 方法 2: 减小 batch size
--set data.train.batch_size=16

# 方法 3: 使用混合精度
--set engine.precision.mixed_precision=true
```

### 如何查看训练过程中的指标曲线

训练结束后会自动在 `outputs/<name>/analysis/training_curves.png` 生成曲线。也可以在训练过程中直接打开该文件（每轮 eval 后更新一次）。

### 如何生成检测的 PR 曲线

```python
from elysia_cv.metrics.iou import MeanAP
from elysia_cv.utils.visualization import plot_pr_curves

# metric.compute() 之后
pr_data = metric.get_pr_data(iou_thr=0.5)
plot_pr_curves(pr_data, "analysis/pr_curves.png", class_names=coco_names)
```