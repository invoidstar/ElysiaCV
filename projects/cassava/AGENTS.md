# projects/cassava — 木薯叶片病害细粒度分类

> **项目侧文档**。本目录描述 Cassava 分类实验，实际代码分布在工程侧的多个位置。
> 类似于 ElysiaRobot 与 mstvla4resta 的关系：框架代码在 `src/elysia_cv/`，项目实验配置、脚本、分析工具聚合在此处记录。

---

## 项目概况

- **任务**：5 类木薯叶片病害细粒度分类
- **数据集**：17,117 train / 4,280 val，严重类别不平衡（CMD = 61.5%）
- **损失函数**：FocalLoss (α=0.25, γ=2.0)
- **框架**：ElysiaCV

---

## 项目文件分布

```
工程侧 (src/elysia_cv/)                    项目侧 (关联到此项目)
──────────────────────────────────────    ─────────────────────────────────
models/classification/cassava/            configs/
├── plugin.py    ← 模型注册 (7 个 spec)      ├── datasets/cassava.yaml
└── runtime.py   ← CassavaModel + loss        ├── models/mlp.yaml .. vit_l16.yaml
                                               └── experiments/cassava_*.yaml (7 个)

training/engine.py                        scripts/
training/loop.py                          └── run_cassava_experiments.sh
training/hooks/                           
                                           tools/
                                           └── analyze.py
                                           
                                           projects/cassava/
                                           ├── AGENTS.md   ← 你在这里
                                           └── tools/analyze.py
```

---

## 启动方式

```bash
# 单模型（统一入口：根目录 run_experiment.sh）
bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --gpu 6

# 带参数覆盖
bash run_experiment.sh configs/experiments/cassava_mlp.yaml --gpu 6 \
    --set schedule.steps=100 data.train.batch_size=8

# 全量对比（7 模型依次跑）
bash scripts/run_cassava_experiments.sh

# 训练后分析
python tools/analyze.py configs/experiments/cassava_resnet50.yaml \
    --checkpoint work_dirs/cassava_resnet50/checkpoint_best_Accuracy.pt
```

---

## 模型列表

| 模型 | 配置 | 参数 | Best Acc |
|------|------|------|----------|
| MLP (64×64) | `cassava_mlp.yaml` | ~13M | 63.13% |
| SimpleCNN | `cassava_simple_cnn.yaml` | ~10M | 81.82% |
| ResNet-18 | `cassava_resnet18.yaml` | ~11M | 83.48% |
| ResNet-34 | `cassava_resnet34.yaml` | ~21M | ~72% |
| ResNet-50 | `cassava_resnet50.yaml` | ~25M | 待跑 |
| ViT-B/16 | `cassava_vit_b16.yaml` | ~86M | 待跑 |
| ViT-L/16 | `cassava_vit_l16.yaml` | ~307M | 待跑 |

---

## 添加新模型到本项目

1. 在 `configs/models/` 添加模型配置 YAML
2. 在 `configs/experiments/` 添加实验组合配置（`_base_: [runtime, model, dataset]`）
3. 在 `src/elysia_cv/models/classification/cassava/plugin.py` 注册 ModelSpec
4. 在 `scripts/run_cassava_experiments.sh` 的 EXPERIMENTS 数组添加条目