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
# 分类
python tools/train.py configs/classification/resnet50_imagenet.yaml

# 检测
python tools/train.py configs/detection/faster_rcnn_resnet50_coco.yaml

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
    --resume ./work_dirs/checkpoint_step_10000.pt
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
model:      # 模型定义 (type + backbone + neck + head)
data:       # 数据源 + 预处理
  train:
    dataset_root: /data/imagenet/train
    batch_size: 256
    transforms: [...]
  val:
    dataset_root: /data/imagenet/val
    batch_size: 64
    transforms: [...]
engine:     # 训练策略 (DDP / precision / compile / optimization)
optim:      # 优化器 (type + params)
schedule:   # 学习率策略 (type + steps + warmup)
tracking:   # 日志 / W&B
eval:       # 评测频率 + 指标
```

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