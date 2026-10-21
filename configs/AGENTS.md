# AGENTS.md — configs/

> 本目录通过 `_base_` 继承实现模型 + 数据集的分离组合式配置。
> 最后更新：2026-10-21

---

## 目录约定

```
configs/
├── _base_/
│   └── default_runtime.yaml      # 全局默认值（engine, tracking, dataloader）
│
├── models/                       # 纯模型定义
│   └── {name}.yaml               # model + optim + schedule + engine.optimization
│
├── datasets/                     # 纯数据集定义
│   └── {name}.yaml               # data 段（dataset_name, dataset_root, batch_size）
│
├── experiments/                  # 实验 = model + dataset 组合
│   └── {dataset}_{model}.yaml    # _base_: [runtime, model, dataset] + 覆盖项
│
├── classification/               # 分类配置（Cassava 9 模型 + ImageNet）
├── detection/                    # 检测配置（FCOS + YOLO 21 变体）
│   └── _base_/yolo_base.yaml     # YOLO detection base
├── segmentation/                 # 分割配置（FCN-R50/R101 + ADE20K/Cityscapes）
│
├── action_recognition/           # 动作识别配置（16 种视频模型）
│   ├── _base_/video_base.yaml    # 视频训练 base（DecordInit → SampleFrames → DecordDecode → Resize → Crop → Normalize → FormatShape → PackVideoInputs）
│   ├── tsn_resnet50_ucf101.yaml  # TSN-ResNet50
│   ├── tsm_resnet50_ucf101.yaml  # TSM-ResNet50
│   ├── mvitv2_small_ucf101.yaml  # MViTv2
│   ├── r3d18_ucf101.yaml         # R3D-18
│   ├── x3d_xs_ucf101.yaml        # X3D-XS
│   ├── xclip_vitb16_ucf101.yaml  # X-CLIP ViT-B/16
│   ├── st_adapter_vitb16_ucf101.yaml  # ST-Adapter ViT-B/16
│   └── uniformerv2_ucf101.yaml   # UniFormerV2
│
└── README.md                     # 配置索引
```

---

## 组合模型

每个实验配置文件通过 `_base_` 列表引用多类配置，从左到右合并：

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
    batch_size: 32                   # Schema B: 平铺键（config loader 自动 normalize）
```

**优先级**：子节点 > 后引用的 `_base_` > 先引用的 `_base_`

---

## Config Schema（统一 Schema B）

所有配置统一使用 **Schema B 平铺键格式**。Config loader (`_normalize_config_schema`) 自动将：
- `data.train.loader.batch_size` → `data.train.batch_size`
- `data.train.loader.num_workers` → `data.train.num_workers`

向后兼容旧格式（`loader:` 子键会自动提升为平铺键）。

```yaml
# Schema B（推荐写法）
data:
  train:
    dataset_name: VIDEO
    dataset_root: /data/ucf101
    format: ucf101
    split: train
    batch_size: 16
    num_workers: 4
    pin_memory: true
    drop_last: false
    transforms:
      - type: DecordInit
      - type: SampleFrames
        num_frames: 8
      - type: DecordDecode
      - type: VideoResize
        scale: 224
      - type: VideoNormalize
        mean: [0.45, 0.45, 0.45]
        std: [0.225, 0.225, 0.225]
      - type: FormatShape
        input_format: NCHW
      - type: PackVideoInputs
```

---

## 添加新配置

### 新模型

创建 `configs/models/my_model.yaml`，必须包含：
- `model` — 架构定义（`type`, `backbone`, `num_classes`, `pretrained`, `dropout`）
- `optim` — 优化器类型和参数
- `schedule` — 学习率调度
- `engine.optimization` — gradient_accumulation_steps、grad_clip_norm

### 新数据集

创建 `configs/datasets/my_data.yaml`，必须包含：
- `data.train` — dataset_name、dataset_root、split、batch_size、transforms
- `data.val` — 同上

### 新增视频模型

✅ 只需在 `model_builder.py` 的 `_MODEL_REGISTRY` 中添加**一行**：

```python
_register("my_video_model", "elysia_cv.models.action_recognition.my_model.runtime", family="video")
```

无需修改 `cli.py`、`engine.py` 或任何其他文件。

### 新实验

创建 `configs/experiments/my_exp.yaml`，`_base_` 引用 model + dataset，覆盖 `run` 和 `data.train.batch_size`。

---

## 注意

- `dataset_root` 使用绝对路径，如 `/data/hdd3/chuqi/data/ucf101/`
- `output_dir` 使用相对路径 `./outputs/checkpoints/...`（运行时从 CWD 解析）
- transforms 在配置中显式定义（不在代码中 hardcode）
- 视频数据集额外需要 `format` 字段（`ucf101` / `kinetics`）和视频专属 transforms（`DecordInit`, `SampleFrames`, `DecordDecode`, `VideoResize`, `VideoRandomCrop`, `VideoNormalize`, `FormatShape`, `PackVideoInputs`）
- Config loader 会自动 normalize `loader.` 子键为平铺键，推荐直接写平铺格式