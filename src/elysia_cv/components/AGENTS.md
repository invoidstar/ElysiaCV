# AGENTS.md — components/

> 本模块存放可复用的 Backbone、Neck、Head 组件。所有组件通过 Registry 注册。

---

## 目录约定

```
components/
├── backbones/    # 特征提取器（ResNet, ViT, MLP, SimpleCNN）
├── necks/        # 特征金字塔 / 中间层（FPN, IdentityNeck）
└── heads/        # 任务头（LinearClsHead, DetHead, SegHead）
```

---

## 添加新组件

**步骤 1**：在对应子目录创建 `my_component.py`

```python
import torch.nn as nn
from elysia_cv.core.registry import BACKBONES

@BACKBONES.register_module()
class MyBackbone(nn.Module):
    """你的 backbone 实现"""
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x):
        ...
        return x
```

**步骤 2**：在 `src/elysia_cv/__init__.py` 的 `_register_default_components()` 中添加 import：

```python
import elysia_cv.components.backbones.my_module  # noqa: F401
```

**你不可以**：
- 在 component 中 import `models/` 或 `training/`
- 硬编码设备（`cuda:0`）
- 在 `__init__` 中做前向计算

**你必须**：
- 使用对应的 Registry 装饰器
- 所有构造参数允许通过 `**kwargs` 透传
- 文档字符串写明参数含义