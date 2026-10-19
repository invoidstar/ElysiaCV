# AGENTS.md — configs/

> 本目录通过 `_base_` 继承实现模型 + 数据集的分离组合式配置。

---

## 目录约定

```
configs/
├── _base_/
│   └── default_runtime.yaml   # 全局默认值（engine, tracking, dataloader）
│
├── models/                    # 纯模型定义
│   └── {name}.yaml            # model + optim + schedule + engine.optimization
│
├── datasets/                  # 纯数据集定义
│   └── {name}.yaml            # data 段（dataset_name, dataset_root, loader）
│
├── experiments/               # 实验 = model + dataset 组合
│   └── {dataset}_{model}.yaml # _base_: [runtime, model, dataset] + 覆盖项
│
└── classification/            # （旧格式，仍可用作兼容）
    └── cassava_{model}.yaml
```

---

## 组合模型

每个实验配置文件通过 `_base_` 列表引用 3 类配置，从左到右合并：

```yaml
# configs/experiments/cassava_resnet50.yaml
_base_:
  - ../_base_/default_runtime.yaml   # 1. 引擎、tracking 默认值
  - ../models/resnet50.yaml          # 2. 模型架构 + 优化器 + 调度器
  - ../datasets/cassava.yaml         # 3. 数据集路径 + loader 配置

run:
  name: cassava_resnet50             # 4. 实验级覆盖（最高优先级）
  output_dir: ./outputs/checkpoints/cassava_resnet50

data:
  train:
    loader:
      batch_size: 32
```

**优先级**：子节点 > 后引用的 `_base_` > 先引用的 `_base_`

---

## 添加新配置

### 新模型

创建 `configs/models/my_model.yaml`，必须包含：
- `model` — 架构定义
- `optim` — 优化器类型和参数
- `schedule` — 学习率调度
- `engine.optimization` — gradient_accumulation_steps、grad_clip_norm

### 新数据集

创建 `configs/datasets/my_data.yaml`，必须包含：
- `data.train` — dataset_name、dataset_root、split、loader
- `data.val` — 同上

### 新实验

创建 `configs/experiments/my_exp.yaml`，`_base_` 引用 model + dataset，覆盖 `run` 和 `data.train.loader.batch_size`。

---

## 注意

- `dataset_root` 使用绝对路径 `/data/hdd3/chuqi/data/...`
- `output_dir` 使用相对路径 `./outputs/checkpoints/...`（运行时从 CWD 解析）
- transforms 由 `_build_dataloaders()` 根据 backbone 类型自动生成，不在配置中定义