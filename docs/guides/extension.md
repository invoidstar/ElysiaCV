# 扩展指南

> 状态：Active（长期有效） | 最后更新：2026-10

---

## 新增模型

新增模型只需创建两个文件：`plugin.py` + `runtime.py`，无需修改任何框架代码。

### 目录结构

```
src/elysia_cv/models/
└── {task_group}/              # classification / detection / segmentation
    └── {model_name}/           # 模型目录
        ├── __init__.py         # 空文件或导入
        ├── plugin.py           # 必须 — 导出 MODEL_PLUGIN
        ├── runtime.py          # 必须 — build_model + forward_train + forward_test
        └── native_source/      # 可选 — 模型内部实现
            └── modeling.py
```

### 步骤 1: 创建 plugin.py

```python
"""MyModel plugin — MODEL_PLUGIN for auto-discovery."""

from elysia_cv.core.api_version import CORE_API_VERSION
from elysia_cv.core.plugin import SimpleModelPlugin
from elysia_cv.core.readiness import PluginReadinessLevel
from elysia_cv.models.contract import ModelSpec

MODEL_PLUGIN = SimpleModelPlugin(
    name="my_model",                      # 全局唯一名
    task_type="classification",           # classification / detection / segmentation
    api_version=CORE_API_VERSION,
    readiness_level=PluginReadinessLevel.trainable,
    model_specs=[
        ModelSpec(
            name="my_model_v1",           # 规格名
            task_type="classification",
            backbone="MyBackbone",
            head="MyHead",
            input_size=(224, 224),
            extra={"detail": "..."},      # 额外元数据
        ),
    ],
)
```

### 步骤 2: 创建 runtime.py

```python
"""MyModel runtime — build_model / forward_train / forward_test."""

from typing import Any
import torch
import torch.nn as nn

from elysia_cv.core.registry import BACKBONES, HEADS, LOSSES


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build model from config dict.

    cfg["model"] 包含 backbone / head 等配置。
    """
    model_cfg = cfg.get("model", cfg)

    # 从 registry 构建 backbone
    backbone_cfg = dict(model_cfg.get("backbone", {}))
    backbone = BACKBONES.build({"type": "MyBackbone", **backbone_cfg})

    # 从 registry 构建 head
    head_cfg = dict(model_cfg.get("head", {}))
    head = HEADS.build({"type": "MyHead", **head_cfg})

    model = MyModel(backbone=backbone, head=head)

    # 加载预训练权重
    pretrained = model_cfg.get("pretrained_path")
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state, strict=False)

    return model


class MyModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def forward_train(model, batch, *, device=None):
    """训练前向。

    必须返回 dict，至少包含 "loss" key。
    可选包含 "losses" (dict) 和 "metrics" (dict) 用于日志。
    """
    images = batch["inputs"]
    labels = batch.get("gt_label")
    if device is not None:
        images = images.to(device, non_blocking=True)
        if labels is not None:
            labels = labels.to(device, non_blocking=True)

    outputs = model(images)
    logits = outputs["logits"]

    loss_fn = LOSSES.build({"type": "CrossEntropyLoss"})
    loss = loss_fn(logits, labels)

    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean()

    return {
        "loss": loss,
        "losses": {"ce": loss.detach()},
        "metrics": {"acc": acc.detach()},
    }


def forward_test(model, batch, *, device=None):
    """评测前向。"""
    images = batch["inputs"]
    if device is not None:
        images = images.to(device, non_blocking=True)
    with torch.no_grad():
        outputs = model(images)
    return {"logits": outputs["logits"].detach()}
```

### 步骤 3: 验证

```bash
# 确认自动发现
python -c "
from elysia_cv.models.registry import ModelRegistry
reg = ModelRegistry.discover()
print('my_model' in reg)  # → True
"

# 运行测试
pytest tests/ -v -k "my_model"
```

### 步骤 4: 创建配置文件

ElysiaCV 采用模型 + 数据集分离组合式配置：

```yaml
# 1) 模型配置 — configs/models/my_model.yaml
model:
  type: my_model
  backbone:
    type: MyBackbone
  head:
    type: LinearClsHead
    num_classes: 5
    in_channels: 512

optim:
  type: SGD
  params:
    lr: 0.01
    momentum: 0.9

schedule:
  type: CosineAnnealing
  steps: 16050
  base_lr: 0.01
  min_lr: 0.00001
  warmup_steps: 500
```

```yaml
# 2) 数据集配置 — configs/datasets/my_data.yaml
data:
  train:
    dataset_name: my_dataset
    dataset_root: /data/my_dataset
    split: train
    loader:
      num_workers: 4
      pin_memory: true
```

```yaml
# 3) 实验组合 — configs/experiments/my_experiment.yaml
_base_:
  - ../_base_/default_runtime.yaml
  - ../models/my_model.yaml
  - ../datasets/my_data.yaml

run:
  name: my_experiment
  output_dir: ./outputs/checkpoints/my_experiment

data:
  train:
    dataset_root: /data/imagenet/train
    batch_size: 32
    transforms:
      - type: Resize
        size: 224
      - type: Normalize
      - type: PackInputs
  val:
    dataset_root: /data/imagenet/val
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
  save_best: Accuracy
  save_final: true
  metrics:
    - Accuracy
```

---

## 新增 Backbone

```python
# src/elysia_cv/components/backbones/my_backbone.py
from elysia_cv.core.registry import BACKBONES
import torch.nn as nn

@BACKBONES.register_module()
class MyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        return self.conv(x)
```

然后在 `components/backbones/__init__.py` 中导入触发注册：

```python
from elysia_cv.components.backbones.my_backbone import MyBackbone  # noqa: F401
```

---

## 新增 Loss

```python
# src/elysia_cv/losses/my_loss.py
from elysia_cv.core.registry import LOSSES
import torch.nn as nn

@LOSSES.register_module()
def MyLoss(weight: float = 1.0, **kwargs) -> nn.Module:
    return nn.MSELoss()
```

然后在 `losses/__init__.py` 中导入触发注册。

---

## 新增 Metric

```python
# src/elysia_cv/metrics/my_metric.py
from elysia_cv.core.registry import METRICS
import torch.nn as nn

@METRICS.register_module()
class MyMetric(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, pred, target):
        self.sum += (pred == target).sum().item()
        self.count += target.numel()

    def compute(self):
        return {"my_metric": self.sum / max(self.count, 1)}
```

然后在 `metrics/__init__.py` 中导入触发注册。

---

## 新增 Transform

```python
# src/elysia_cv/data/transforms/my_transform.py
from elysia_cv.core.registry import TRANSFORMS

@TRANSFORMS.register_module()
class MyTransform:
    def __init__(self, param: int = 42, **kwargs):
        self.param = param

    def __call__(self, results):
        results["my_key"] = self.param
        return results
```

---

## 新增 Hook

```python
# src/elysia_cv/my_hooks/custom.py
from elysia_cv.training.hooks.base import Hook

class CustomHook(Hook):
    def after_step(self, loop):
        if loop.state.global_step % 1000 == 0:
            print(f"Custom hook @ step {loop.state.global_step}")
```

在 TrainingEngine 中注册：

```python
hooks.register(CustomHook())
```

---

## 社区模型接入

将社区模型放在 `models/{task}/community/{name}/` 下：

```
src/elysia_cv/models/detection/community/
└── yolo/
    ├── plugin.py          # MODEL_PLUGIN
    └── runtime.py         # 桥接 ultralytics YOLO
```

`plugin.py` 与内置模型完全相同的写法，`runtime.py` 内部可以导入任何第三方包。框架不关心内部实现。