# ElysiaCV Cassava Leaf Disease Classification — 完整实验报告

> 数据集: Cassava Leaf Disease Classification  
> 类别: CBB / CBSD / CGM / CMD / Healthy (5 分类)  
> 训练/验证: 17,118 / 4,280 张  
> 公平预算: 16,050 步, GTX 3090 24G × 单卡  

---

## 一、实验结果总览

| # | 模型 | Backbone | Params | Optim | LR (base) | Eff. Batch | Img | Best Acc | Loss | 耗时 |
|---|------|----------|--------|-------|-----------|------------|-----|----------|------|------|
| 1 | MLP | Flatten+FC | 13.1M | AdamW | 1e-3 | 32 | 64×64 | 63.13% | 0.1503 | ~0.3h |
| 2 | SimpleCNN | 5-stage CNN | 9.7M | SGD | 1e-2 | 32 | 448×448 | 81.82% | 0.0742 | ~0.5h |
| 3 | ResNet18 | ResNet-18 | 11.2M | SGD | 1e-2 | 32 | 448×448 | 83.48% | 0.0607 | ~0.8h |
| 4 | ResNet34 | ResNet-34 | 21.3M | SGD | 1e-2 | 32 | 448×448 | 81.45% | 0.0661 | ~0.9h |
| 5 | ResNet50 | ResNet-50 | 23.5M | SGD | 1e-2 | 32 | 448×448 | 67.15% | 0.1199 | 2.3h |
| 6 | ViT-B/16 | ViT-Base 16 | 86.3M | AdamW | 3e-4 | 32 | 448×448 | 68.81% | 0.1134 | 7.1h |
| 7 | ViT-L/16 | ViT-Large 16 | 303.9M | AdamW | 1e-4 | 32 | 448×448 | 76.45% | 0.0824 | 17.5h |
| 8 | HERBS Swin-Tiny | Swin-T+HERBS | 27.5M | AdamW | 1e-4 | 8 | 224×224 | 82.52% | 0.0898 | 1.2h |
| 9 | HERBS Swin-Base | Swin-B@384+HERBS | 86.9M | AdamW | 3e-5 | 16 | 384×384 | **89.11%** 🥇 | 0.0767 | 5.7h |
| 10 | HERBS Swin-Tiny (重训) | Swin-T+HERBS | 27.5M | AdamW | 1e-4 | 8 | 224×224 | 87.69% | 0.0520 | 1.1h |
| 11 | HERBS Swin-Base (重训) | Swin-B@384+HERBS | 86.9M | AdamW | 3e-5 | 16 | 384×384 | **89.46%** 👑 | 0.0577 | 5.9h |

---

## 二、精度排名与收敛曲线

### 2.1 最终排名

```
Rank  Model                         Accuracy    参数量
────  ────────────────────────────  ────────    ─────
 👑   HERBS Swin-Base (retrain)     89.46%      86.9M
 🥈   HERBS Swin-Base (v1)          89.11%      86.9M
 🥉   HERBS Swin-Tiny (retrain)     87.69%      27.5M
  4   ResNet18                      83.48%      11.2M
  5   HERBS Swin-Tiny (v1)          82.52%      27.5M
  6   SimpleCNN                     81.82%      9.7M
  7   ResNet34                      81.45%      21.3M
  8   ViT-L/16                      76.45%      303.9M
  9   ViT-B/16                      68.81%      86.3M
 10   ResNet50                      67.15%      23.5M
 11   MLP                           63.13%      13.1M
```

### 2.2 精度 vs 参数量散点图

```
89% ┤                                   ● HERBS-Base (86.9M) 👑
    │
87% ┤                        ● HERBS-Tiny (27.5M)
    │
83% ┤         ● ResNet18 (11.2M)
82% ┤    ● SimpleCNN (9.7M) ● HERBS-Tiny v1 (27.5M)
81% ┤         ● ResNet34 (21.3M)
    │
76% ┤                                          ● ViT-L (304M)
    │
69% ┤                              ● ViT-B (86.3M)
67% ┤                  ● ResNet50 (23.5M)
63% ┤  ● MLP (13.1M)
    │
    └─────────────────────────────────────────────────→ 参数量
    10M  30M  100M  300M
```

---

## 三、各模型详细分析

### 3.1 MLP (63.13%)

| 配置 | 值 |
|------|-----|
| Backbone | Flatten → [1024, 512]，dropout=0.3 |
| Head | Linear(512→5)，dropout=0.2 |
| 图像尺寸 | 64×64，无数据增强 |
| 优化器 | AdamW，lr=1e-3，wd=1e-4 |
| Scheduler | Cosine，warmup=300，min_lr=1e-5 |

**训练曲线**:
```
Step 1k: 61.85% → 5k: 62.64% → 10k: 63.13% → 16k: 63.01%
```

**分析**: 纯 MLP 在 64×64 低分辨率下达到 63%，作为 baseline 合理。收敛快但上限低，受限于无空间归纳偏置。

---

### 3.2 SimpleCNN (81.82%)

| 配置 | 值 |
|------|-----|
| Backbone | 5-stage 纯 CNN，base_channels=64，pool_last |
| Head | Linear(512→5)，dropout=0.2 |
| 图像尺寸 | 448×448，ColorJitter + RandomFlip |
| 优化器 | SGD，lr=1e-2，momentum=0.9 |
| Scheduler | Cosine，warmup=500，min_lr=1e-5 |

**训练曲线**:
```
Step 1k: 67.13% → 4k: 72.57% → 11k: 78.90% → 16k: 81.82% (稳步上升)
```

**分析**: 仅 9.7M 参数达到 81.82%，几乎与 ResNet34 (21.3M) 持平，说明在中等规模数据集上深度未必是决定因素。收敛平稳，未出现过拟合。

---

### 3.3 ResNet18 (83.48%) — CNN 最佳

| 配置 | 值 |
|------|-----|
| Backbone | ResNet-18，pool_last=true |
| Head | Linear(512→5)，dropout=0.2 |
| 图像尺寸 | 448×448，ColorJitter + RandomFlip |
| 优化器 | SGD，lr=1e-2，momentum=0.9 |
| Scheduler | Cosine，warmup=500，min_lr=1e-5 |

**训练曲线**:
```
Step 1k: 62.57% → 8k: 72.00% → 13k: 78.00% → 15k: 83.00% → 16k: 83.48%
```

**分析**: ResNet18 是**最佳 CNN baseline**，11.2M 参数达到 83.48%。比 ResNet34 高 2pp，比 ResNet50 高 16pp。ResNet50 明显欠拟合——SGD+1e-2 的 lr 可能偏向小模型，需要更多步或更小的 lr。

---

### 3.4 ResNet50 (67.15%) — 严重退化

| 配置 | 值 |
|------|-----|
| Backbone | ResNet-50，pool_last=true |
| Head | Linear(2048→5)，dropout=0.2 |
| 优化器 | SGD，lr=1e-2，momentum=0.9 |

**训练曲线** (几乎不动):
```
Step 1k: 61.50% → 4k: 63.62% → 8k: 63.74% → 16k: 67.15%
```

**分析**: ResNet50 在相同超参下严重欠拟合。BottleNeck 残差块 (1×1→3×3→1×1) 更深但需要更精细的学习率调度。SGD lr=1e-2 对 ResNet50 可能偏大，导致优化不稳定。这是典型的"更大模型 ≠ 更好结果"案例。

---

### 3.5 ViT-B/16 (68.81%)

| 配置 | 值 |
|------|-----|
| Backbone | ViT vit_b_16，image_size=448，pretrained=false |
| Head | Linear(768→5)，dropout=0.1 |
| Eff. Batch | 32 (16×GA2) |
| 优化器 | AdamW，lr=3e-4，wd=0.05 |
| Scheduler | Cosine，warmup=1000，min_lr=1e-6 |

**训练曲线** (缓慢爬升):
```
Step 1k: 64.09% → 6k: 65.74% → 11k: 66.66% → 16k: 68.81%
```

**分析**: ViT-B/16 从零训练 (no pretrain)，18K 张图不足以让 86M 参数的 ViT 学到有效表示。从头训练 ViT 在本数据集上几乎不收敛——每一步仅提升约 0.03pp。

---

### 3.6 ViT-L/16 (76.45%)

| 配置 | 值 |
|------|-----|
| Backbone | ViT vit_l_16，image_size=448，pretrained=false |
| Head | Linear(1024→5)，dropout=0.1 |
| Eff. Batch | 32 (4×GA8) |
| 优化器 | AdamW，lr=1e-4，wd=0.05 |
| Scheduler | Cosine，warmup=1500，min_lr=1e-6 |

**训练曲线** (持续上升):
```
Step 1k: 65.70% → 5k: 69.02% → 9k: 72.90% → 13k: 75.12% → 16k: 76.45%
```

**分析**: ViT-L/16 虽然参数爆炸 (304M)，但表现出持续学习能力——每 2K 步提升约 1.5pp。76.45% 远超 ViT-B/16 的 68.81%，说明更大模型确实能学到更多。如果训练 30K+ 步，预计可达 80%+，但成本极高 (17.5h 对应 16K 步)。

---

### 3.7 HERBS Swin-Tiny (v1: 82.52%, retrain: 87.69%)

| 配置 | 值 |
|------|-----|
| Backbone | SwinTransformer swin_tiny，pretrained=true，features_only |
| Neck | HERBSNeck，num_classes=5，fpn_size=384 |
| Neck 组件 | FPN→WeaklySelector→GCNCombiner |
| 图像尺寸 | 224×224，ColorJitter + RandomFlip |
| 优化器 | AdamW，lr=1e-4，wd=0.05 |
| Eff. Batch | 8 |

**v1 训练曲线**:
```
Step 500: 31.80% → 3.5k: 62.31% → 8k: 72.29% → 11k: 79.95% → 14.5k: 82.52%
```

**retrain 训练曲线**:
```
Step 500: 65.93% → 1.5k: 81.59% → 3k: 84.18% → 9k: 87.27% → 15k: 87.69%
```

**分析**: 重训大幅提升 (+5.2pp)。v1 初期极低 (31.8% → 500 步) 说明首轮训练不稳定（可能初始化波动）。重训后收敛速度极快：1.5K 步已达 81.6%。Swin+HERBS 的组合展现了强大的少样本学习能力。

---

### 3.8 HERBS Swin-Base (v1: 89.11%, retrain: 89.46%) 👑

| 配置 | 值 |
|------|-----|
| Backbone | SwinTransformer swin_base_384，pretrained=true，features_only |
| Neck | HERBSNeck，num_classes=5，fpn_size=512 |
| 图像尺寸 | 384×384，ColorJitter + RandomFlip |
| 优化器 | AdamW，lr=3e-5，wd=0.05 |
| Eff. Batch | 16 (batch 4×GA4) |

**v1 训练曲线**:
```
Step 500: 21.89% → 2k: 76.45% → 3.5k: 84.51% → 5k: 87.57% → 14.5k: 89.11%
```

**retrain 训练曲线**:
```
Step 500: 71.87% → 1k: 84.95% → 2k: 87.48% → 7k: 89.23% → 7.5k: 89.46%
```

**分析**: **断层第一**，领先第二名 1.8pp。retrain 仅比 v1 高 0.35pp，说明 Base 模型训练已经非常稳定。仅 2K 步 (约 43 分钟) 即达到 87.5%，收敛速度惊人。优势来源：
1. **预训练权重**：ImageNet-21K 预训练的 Swin-Base 提供了强大的视觉先验
2. **HERBS 机制**：WeaklySelector 的背景抑制 + GCN 的 token 交互克服了大模型在小数据上的过拟合倾向
3. **多尺度 FPN**：融合 4 层特征 (96²/48²/24²/12²)，细粒度病害特征被充分保留

---

## 四、训练策略对比

### 4.1 优化器

| 优化器 | 适用模型 | LR 范围 | 特点 |
|--------|---------|---------|------|
| SGD+momentum | SimpleCNN, ResNet 系列 | 1e-2 | CNN 通用，简单有效 |
| AdamW | MLP, ViT 系列, HERBS 系列 | 1e-4 ~ 1e-3 | Transformer 必备，自适应学习率 |

### 4.2 Scheduler

全部使用 **CosineAnnealing**，min_lr=1e-5 (SGD) 或 1e-6 (AdamW)。

### 4.3 Warmup

| 模型 | Warmup Steps |
|------|-------------|
| MLP | 300 |
| CNN 系列 | 500 |
| ViT-B/16 | 1000 |
| ViT-L/16 | 1500 |
| HERBS 系列 | 500 |

Transformer 类需要更长 warmup 以避免训练初期不稳定。

---

## 五、关键发现

### 5.1 HERBS 机制显著有效

HERBS (Hierarchical Background Suppression) 通过 WeaklySelector + GCNCombiner 实现了背景抑制和高温细化，在小数据集上取得 89.46% 精度的同时避免了过拟合。这是本实验**最大的方法论贡献**。

Swin-Base 参数量 (86.9M) 与 ViT-B/16 (86.3M) 几乎相同，但精度差距达 **20.7pp**——说明**架构设计比参数量更重要**。

### 5.2 ResNet 系列反常

传统认知中 ResNet50 > ResNet34 > ResNet18，但本实验中：
- ResNet18 (83.48%) > ResNet34 (81.45%) > ResNet50 (67.15%)

可能原因：
1. SGD+lr=1e-2 偏向更浅的网络
2. 16,050 步对大模型 ResNet50 来说不够
3. BatchNorm 在小 batch (32) 下对深层网络不够稳定

### 5.3 ViT 从头训练不可行

ViT-B/16 (68.81%) 和 ViT-L/16 (76.45%) 均从头训练，在本数据集上严重逊于有预训练权重的同类 (Swin)。验证了"ViT 需要海量数据或预训练"的共识。

### 5.4 低成本高效率

- **SimpleCNN** 仅 9.7M / 0.5h 即达 81.82%，性价比极高
- **HERBS Swin-Tiny** 27.5M / 1.1h 达 87.69%，效率惊人
- **HERBS Swin-Base** 5.9h 达 89.46%，投入产出比远超 ViT-L/16 (17.5h / 76.45%)

### 5.5 Bug 修复验证

重训 HERBS 前修复了 Swin 骨干网络的 **NHWC/NCHW 维度 Bug**（之前错误将 timm 的 NHWC 输出解析为 NCHW，导致所有 4 层特征映射到相同空间分辨率）。修复后：
- Swin-Tiny: 82.52% → 87.69% (+5.17pp) 
- Swin-Base: 89.11% → 89.46% (+0.35pp)

正确定维后 Tiny 模型收益巨大，Base 因本身强大先验已接近饱和。

---

## 六、GPU 资源统计

| GPU | 模型 | 显存占用 | 训练时长 |
|-----|------|---------|---------|
| GPU 5 | SimpleCNN, ResNet18/34/50 | ~8G | 0.5-2.3h |
| GPU 6 | ViT-B/16, ViT-L/16, MLP | ~20G | 0.3-17.5h |
| GPU 5 | HERBS Swin-Tiny/Base | ~20G | 1.2-5.9h |

总训练时长: ~42 GPU 小时

---

## 七、文件索引

```
outputs/checkpoints/
├── cassava_mlp/                   # MLP baseline (150MB)
├── cassava_simple_cnn/            # SimpleCNN (37MB)
├── cassava_resnet18/              # ResNet18 best CNN (43MB)
├── cassava_resnet34/              # ResNet34 (81MB)
├── cassava_resnet50/              # ResNet50 退化 (90MB)
├── cassava_vit_b16/               # ViT-B/16 欠拟合 (987MB)
├── cassava_vit_l16/               # ViT-L/16 持续学习 (3.4GB)
├── cassava_herbs_swin_tiny/       # HERBS Tiny (334MB)
└── cassava_herbs_swin_base/       # HERBS Base 👑 (1GB)

outputs/heatmaps/
├── all_models_heatmaps.png        # 7 模型 × 5 类 Grad-CAM
├── herbs_swin_base_multistage.png # HERBS Base 4 阶段注意力演进
└── backbone_energy_fixed.png      # Swin 骨干多尺度能量验证

logs/
├── cassava_experiments_20261012_103740/  # 第一轮 CNN/ViT 日志
├── herbs_swin_tiny_20261012_232135.log   # HERBS Tiny v1
├── herbs_swin_base_20261013_121632.log   # HERBS Base v1
├── herbs_swin_tiny_retrain_20261014*.log # Tiny 重训
└── herbs_swin_base_retrain_20261014*.log # Base 重训
```

---

## 八、总结

| 维度 | 最佳模型 | 数值 |
|------|---------|------|
| 绝对精度 | HERBS Swin-Base (retrain) | **89.46%** |
| 性价比 (acc/params) | ResNet18 | 0.75%/M |
| 速度 | SimpleCNN | 81.82% @ 0.5h |
| 样本效率 | HERBS Swin-Base | 87.5% @ 2K 步 |

**HERBS + Swin Transformer** 是本轮实验的明确赢家，在精度、收敛速度、抗过拟合三个维度均显著优于传统 CNN 和纯 ViT 方案。

---

*报告生成时间: 2026-10-14*  
*实验环境: NVIDIA RTX 3090 24G × 单卡, PyTorch + timm*  
*数据集: Cassava Leaf Disease Classification (Kaggle 2019)*