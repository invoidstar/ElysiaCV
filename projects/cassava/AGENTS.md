# AGENTS.md — projects/cassava/

> Cassava 叶片病害细粒度分类实验项目。

---

## 目录

```
projects/cassava/
├── AGENTS.md                  ← 当前文件
├── scripts/
│   ├── run.sh                 ← 单机单卡顺序实验
│   └── ddp_run.sh             ← 单机多卡 (torchrun) 单模型
├── tools/
│   └── analyze.py             ← 训练后分析（混淆矩阵 + GradCAM）
└── configs -> ../../configs/  ← 逻辑符号链接（配置统一在根 configs/ 管理）
```

---

## 启动方式

### 单卡顺序实验（7 个模型依次跑）

```bash
bash projects/cassava/scripts/run.sh
```

### 单机多卡训练（torchrun）

```bash
# 2 卡 ResNet-50
CUDA_VISIBLE_DEVICES=0,1 bash projects/cassava/scripts/ddp_run.sh \
    configs/experiments/cassava_resnet50.yaml

# 4 卡 ViT-B/16
CUDA_VISIBLE_DEVICES=0,1,2,3 bash projects/cassava/scripts/ddp_run.sh \
    configs/experiments/cassava_vit_b16.yaml

# 带参数覆盖
CUDA_VISIBLE_DEVICES=0,1 bash projects/cassava/scripts/ddp_run.sh \
    configs/experiments/cassava_resnet50.yaml \
    --set schedule.steps=8000 optim.params.lr=0.02
```

### 训练后分析

```bash
python projects/cassava/tools/analyze.py \
    configs/experiments/cassava_resnet50.yaml \
    --checkpoint work_dirs/cassava_resnet50/checkpoint_best_Accuracy.pt
```

---

## 数据集

Cassava Leaf Disease Classification:
- 5 类：CBB, CBSD, CGM, CMD, Healthy
- 17,117 train / 4,280 val
- 严重类别不平衡（CMD = 61.5%）
- 使用 FocalLoss (α=0.25, γ=2.0) 处理

---

## 已测试模型

| 模型 | Best Acc | 说明 |
|------|----------|------|
| MLP (64×64) | 63.13% | 简单基线 |
| SimpleCNN | 81.82% | 5 阶段 CNN，无残差 |
| ResNet-18 | 83.48% | 当前最佳 |
| ResNet-34 | ~72% | 震荡大 |
| ResNet-50 | - | 待跑 |
| ViT-B/16 | - | 待跑 |
| ViT-L/16 | - | 待跑 |