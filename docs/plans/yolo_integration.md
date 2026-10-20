# YOLO 系列集成方案

> **状态**：✅ Done · 2026-10-20 · 纯 PyTorch 自实现，零 ultralytics 依赖 · 140 tests pass

---

## 1. 目标

在 ElysiaCV 中集成以下检测模型，全部通过纯 PyTorch 自实现：

| 模型 | 版本 | 发表 | 核心特点 |
|------|------|------|---------|
| **YOLOv10** | n/s/m/l/x | 2024 | NMS-free 双头训练 |
| **YOLOv11** | n/s/m/l/x | 2024 | C3k2 模块 + 增强检测头 |
| **YOLOv12** | n/s/m/l/x | 2025 | A2C2f 注意力中心架构 |
| **YOLO26** | n/s/m/l/x | 2026 | 最新版，多数据集 benchmark 验证 |
| **HDMS-YOLO** | — | 2025 | 复杂农田杂草专用：SimSPPF + DCNv3 + EMA |

---

## 2. 为什么不依赖 ultralytics？

| 维度 | 依赖 ultralytics | 纯 PyTorch 自实现 |
|------|:---:|:---:|
| 训练引擎 | `.train()` 黑盒 | 完全复用 ElysiaCV TrainerLoop + HookBus |
| 配置系统 | 两套配置桥接 | 原生 YAML `_base_` 继承 |
| EMA/SWA/EarlyStopping/DDP | 无法使用 | 全部现成可用 |
| 架构修改 | 受限 | 完全可控 |
| License | AGPL-3.0 传染性 | 自主可控 |
| 长期维护 | 被动跟随上游更新 | 自主维护 |

---

## 3. 文件结构

```
src/elysia_cv/models/detection/yolo/
├── __init__.py
├── plugin.py              # @register_model("yolo")
├── runtime.py             # build_model / forward_train / forward_test
├── model.py               # YOLO 模型组装

├── backbone/
│   ├── __init__.py
│   ├── csp_darknet.py     # Conv/Bottleneck/C2f/C3k2/SPPF/CSPDarkNet
│   └── a2c2f.py           # YOLOv12 A2C2f 注意力中心模块

├── neck/
│   ├── __init__.py
│   └── pan.py             # PAN-FPN + 版本开关

├── head/
│   ├── __init__.py
│   ├── detect_head.py     # YOLOv8/v11/v12/v26 解耦检测头
│   └── v10_head.py        # YOLOv10 NMS-free 双头 (one2many + one2one)

├── losses/
│   ├── __init__.py
│   ├── v8_loss.py         # CIoU + DFL + BCE + TaskAlignedAssigner
│   └── v10_loss.py        # YOLOv10 NMS-free 双重 loss

├── postprocess/
│   ├── __init__.py
│   ├── nms.py             # NMS / Soft-NMS
│   └── v10_post.py        # YOLOv10 NMS-free 后处理

├── variants.py            # 20 变体 depth/width 配置表
├── weights.py             # .pt 权重加载 + 键名映射

└── hdms_yolo.py           # HDMS-YOLO: SimSPPF + DCNv3 + EMA Attention
```

---

## 4. 架构设计

### 4.1 各版本模块差异

```
                  Backbone        Neck            Head          NMS        Loss
YOLOv10    ────   C2f          ── PAN-FPN     ── 双头(O2M+O2O) ── 无      ── CIoU×2
YOLOv11    ────   C3k2         ── PAN+C3k2    ── 解耦头       ── NMS     ── CIoU
YOLOv12    ────   A2C2f        ── PAN+A2C2f   ── 解耦+注意    ── NMS     ── CIoU
YOLO26     ────   C3k2改进     ── PAN改进      ── 解耦改进     ── NMS     ── CIoU
HDMS-YOLO  ────   C2f+DCNv3    ── PAN+EMA     ── 解耦头       ── NMS     ── CIoU
```

### 4.2 共享组件列表

| 组件 | 文件 | 行数 | 复用方 |
|------|------|------|--------|
| Conv-BN-SiLU | `csp_darknet.py` | ~40 | 全部 |
| Bottleneck | `csp_darknet.py` | ~30 | v10/v11/v12/v26/HDMS |
| C2f (CSP瓶颈) | `csp_darknet.py` | ~40 | v10 |
| C3k2 (大核CSP) | `csp_darknet.py` | ~50 | v11/v12/v26 |
| SPPF | `csp_darknet.py` | ~30 | 全部 |
| CSPDarkNet | `csp_darknet.py` | ~80 | 全部 (按 variant 调整 depth/width) |
| A2C2f | `a2c2f.py` | ~100 | v12 |
| PAN-FPN | `pan.py` | ~150 | 全部 |
| DetectHead | `detect_head.py` | ~150 | v11/v12/v26/HDMS |
| V10Head | `v10_head.py` | ~200 | v10 |
| NMS | `nms.py` | ~80 | v11/v12/v26/HDMS |
| CIoU Loss | `v8_loss.py` | ~150 | 全部 |
| TaskAlignedAssigner | `v8_loss.py` | ~100 | 全部 |

**核心原则**：70%+ 代码共享，版本差异仅在 `model.py` 的 `_build_backbone/neck/head` 中通过版本号 switch。

---

## 5. 变体配置

### 5.1 骨干缩放参数

```python
# variants.py
YOLO_VARIANTS = {
    # format: (depth_multiple, width_multiple, max_channels)
    "yolov10n": (0.33, 0.25, 1024),
    "yolov10s": (0.33, 0.50, 1024),
    "yolov10m": (0.67, 0.75, 1024),     # v10 无 l/x，m 最大
    "yolov11n": (0.50, 0.25, 1024),
    "yolov11s": (0.50, 0.50, 1024),
    "yolov11m": (0.50, 0.75, 1024),
    "yolov11l": (1.00, 1.00, 1024),
    "yolov11x": (1.00, 1.25, 1024),
    "yolov12n": (0.50, 0.25, 1024),
    "yolov12s": (0.50, 0.50, 1024),
    "yolov12m": (0.50, 0.75, 1024),
    "yolov12l": (1.00, 1.00, 1024),
    "yolov12x": (1.00, 1.25, 1024),
    "yolov26n": (0.33, 0.25, 1024),
    "yolov26s": (0.33, 0.50, 1024),
    "yolov26m": (0.67, 0.75, 1024),
    "yolov26l": (1.00, 1.00, 1024),
    "yolov26x": (1.00, 1.25, 1024),
}
```

### 5.2 各变体 COCO mAP（参考）

| 变体 | 参数量 | mAP@50:95 | 推理延迟 (T4) |
|------|--------|-----------|---------------|
| yolov10n | 2.3M | 38.5% | 1.8ms |
| yolov11n | 2.6M | 39.5% | 1.5ms |
| yolov12n | 2.6M | 40.6% | 1.6ms |
| yolov26n | 2.4M | 42.0% | 1.4ms |
| yolov10s | 7.2M | 46.1% | 2.5ms |
| yolov11s | 9.4M | 47.0% | 2.3ms |
| yolov12s | 9.3M | 48.0% | 2.5ms |
| yolov26s | 9.5M | 48.1% | 2.2ms |

---

## 6. 训练集成

### 6.1 配置文件

```yaml
# configs/detection/yolo11n_coco.yaml
_base_: "../_base_/default_runtime.yaml"

model:
  type: yolo
  variant: yolov11n
  pretrained: true
  num_classes: 80

data:
  train:
    type: CocoDetectionDataset
    root: /data/coco
    transforms:
      - {type: LoadImageFromFile}
      - {type: Mosaic, output_size: [640, 640], prob: 0.5}
      - {type: MixUp, alpha: 8.0, prob: 0.1}
      - {type: RandomFlip, prob: 0.5}
      - {type: Resize, size: [640, 640]}
      - {type: Normalize}
      - {type: PackDetInputs}

engine:
  optim:
    gradient_accumulation_steps: 1
    ema_enable: true
    ema_decay: 0.9999
    swa_enable: false

optim:
  type: AdamW
  params:
    lr: 0.001
    weight_decay: 0.0005

schedule:
  type: CosineAnnealing
  params:
    T_max: 300
    eta_min: 1e-5

eval:
  eval_freq: 5
  metrics: [MeanAP]
  early_stop_enable: false
```

### 6.2 训练入口

```python
# runtime.py

@register_model("yolo")
def build_model(cfg: dict) -> nn.Module:
    variant = cfg["model"]["variant"]
    num_classes = cfg["model"]["num_classes"]
    arch = YOLO_VARIANTS[variant]
    model = YOLODetector(arch, num_classes)
    if cfg["model"].get("pretrained"):
        load_pretrained_weights(model, variant)
    return model

def forward_train(model, batch, *, device=None, loss_weights=None):
    # 完全复用 ElysiaCV TrainerLoop
    images = batch["inputs"].to(device)
    preds = model(images)
    gt = {"bboxes": batch["gt_bboxes"], "labels": batch["gt_labels"]}
    loss_dict = model.compute_loss(preds, gt)
    total = sum(loss_dict.values())
    return {"loss": total, "losses": loss_dict, "metrics": {}}

def forward_test(model, batch, *, device=None):
    images = batch["inputs"].to(device)
    preds = model(images)
    results = model.postprocess(preds)
    return {"results": results}  # COCO format
```

### 6.3 训练命令

```bash
# 单卡
python tools/train.py configs/detection/yolo11n_coco.yaml

# 多卡 DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    tools/train.py configs/detection/yolo11n_coco.yaml

# 农业杂草检测
python tools/train.py configs/detection/yolov26s_weed.yaml \
    --set model.num_classes=74 data.train.root=/data/datasets/CropAndWeed
```

---

## 7. 权重加载

从 ultralytics 官方 `.pt` 文件加载预训练权重：

```python
# weights.py

def load_pretrained_weights(model, variant: str):
    """从 ultralytics .pt 加载权重并做键名映射"""
    url = WEIGHT_URLS[variant]       # GitHub releases 直链
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    # 键名映射: ultralytics 命名 → ElysiaCV 命名
    remapped = {}
    for k, v in state.items():
        new_k = remap_key(k, variant)
        if new_k:
            remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    logger.info(f"Loaded pretrained weights for {variant}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected")
```

---

## 8. HDMS-YOLO 集成

HDMS-YOLO 是对 YOLOv11 的三处改进，可拆分为独立模块：

| 改进 | 说明 | 集成方式 |
|------|------|---------|
| **SimSPPF** | SiLU→ReLU 池化金字塔 | `backbone/csp_darknet.py` 中添加 `SPPF(replace_silu_with_relu=True)` |
| **C2f_DCNv3** | 可变形卷积替换 C2f 中的标准卷积 | `backbone/csp_darknet.py` 中添加 `C2f_DCNv3` 模块 |
| **EMA Attention** | Efficient Multi-Scale Attention | `neck/pan.py` 中注入 EMA 模块 |

```python
# hdms_yolo.py
@register_model("hdms_yolo")
def build_model(cfg):
    arch = YOLO_VARIANTS[cfg["model"]["base_variant"]]  # 基于 yolov11 系列
    model = YOLODetector(arch, cfg["model"]["num_classes"])
    # 注入 HDMS 改进
    inject_simsppf(model.backbone)
    inject_dcnv3(model.backbone)
    inject_ema_attention(model.neck)
    return model
```

---

## 9. 实施计划

| 阶段 | 内容 | 预计工作量 | 状态 |
|------|------|:---:|:---:|
| **P1** | 共享组件: Conv/C2f/C3k2/SPPF/CSPDarkNet/PAN/DetectHead | 2 天 | ✅ |
| **P2** | YOLOv11n 组装 + loss + 权重加载 + 训练跑通 | 2 天 | ✅ |
| **P3** | YOLOv10 NMS-free + YOLOv12 A2C2f + YOLO26 适配 | 2 天 | ✅ |
| **P4** | 20 变体配置 + 预训练权重下载 + 测试 | 1 天 | ✅ |
| **P5** | HDMS-YOLO 集成 + 21 训练配置 + 权重端到端验证 | 1 天 | ✅ |
| **总计** | | **8 天** | ✅ Done |

### 9.1 最终交付物

| 类别 | 数量 | 详情 |
|------|:---:|------|
| YOLO 源码文件 | 18 | model/variants/runtime/plugin + backbone/neck/head + losses/postprocess + weights/hdms |
| 配置 YAML | 22 | `_base_/yolo_base.yaml` + 18 standard + 3 HDMS |
| 注册变体 | 21 | 18 standard (v10/v11/v12/v26) + 3 HDMS |
| 预训练 URL | 18 | v10(3) + v11(5) + v12(5) + v26(5, 兼容 v11 .pt) |
| 测试覆盖 | 140 | 全量回归 140 passed |

### 9.2 版本能力矩阵（最终）

| 版本 | 前向 | Loss | Grad | NMS | 权重URL | 配置 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLOv10 (n/s/m) | ✅ | ✅ dual | ✅ | ✅ NMS-free | ✅ | ✅ |
| YOLOv11 (n/s/m/l/x) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLOv12 (n/s/m/l/x) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLO26 (n/s/m/l/x) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HDMS (n/s/m) | ✅ | ✅ | ✅ | ✅ | N/A | ✅ |

### 9.3 权重加载机制

- **Ultralytics-free**：通过 `sys.modules` 注入 `StubModule`，无需安装 ultralytics 即可从 `.pt` 提取权重
- **三通道映射**：Direct（键名直映射）→ Sequential（位置匹配）→ Shape（形状匹配）
- **缓存路径**：`/data/hdd3/chuqi/pretrained/`

---

## 10. 与现有架构的兼容性

| 现有模块 | YOLO 是否复用 | 说明 |
|---------|:---:|------|
| TrainerLoop | ✅ | forward_train 标准接口 |
| HookBus | ✅ | Logger/Checkpoint/Eval/LR/EMA/SWA/EarlyStopping 全部可用 |
| DDP/FSDP | ✅ | `wrap_ddp` / `wrap_fsdp` |
| CLI (tools/train.py) | ✅ | `elysia train config.yaml --resume` |
| MetricHistory | ✅ | 训练曲线自动生成 |
| Data Pipeline | ✅ | COCO/ADE20K/Cityscapes/CropAndWeed 全部可用 |
| Mosaic/MixUp | ✅ | ElysiaCV 现有 batch 变换管线 |