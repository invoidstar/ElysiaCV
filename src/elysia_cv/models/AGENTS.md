# AGENTS.md — models/

> 本模块存放任务级模型插件。每个任务一个子目录，插件自注册，运行时负责前向计算。

---

## 目录约定

```
models/
├── contract.py         # ModelPlugin 抽象契约
├── registry.py         # MODEL_REGISTRY
├── classification/     # 分类任务
│   └── cassava/        # 示例：Cassava 病害分类
│       ├── plugin.py   #   @register_model 入口
│       └── runtime.py  #   build_model(), forward_train(), forward_test()
├── detection/          # 检测任务
├── segmentation/       # 分割任务
└── dummy/              # CI 测试用虚拟模型
```

---

## 添加新模型插件

每个新模型需要两个文件：

### `plugin.py` — 模型注册

```python
from elysia_cv.models.registry import register_model
from elysia_cv.core.plugin import PluginBase

@register_model("my_model")
class MyModelPlugin(PluginBase):
    task_type = "classification"

    @classmethod
    def build_from_config(cls, cfg):
        from .runtime import build_model
        return build_model(cfg)
```

### `runtime.py` — 前向计算逻辑

```python
def build_model(model_cfg: dict) -> nn.Module:
    """从配置构建模型。"""
    ...

def forward_train(model, batch: dict) -> dict:
    """训练前向：返回 {"loss": ...}"""
    ...

def forward_test(model, batch: dict) -> dict:
    """测试前向：返回 {"logits": ...}"""
    ...
```

---

## 所有权约定

| 谁负责 | 什么 |
|--------|------|
| `runtime.py` | 前向计算、loss 组装、metric 计算 |
| `training/loop.py` | backward、optimizer.step、scheduler.step |
| `cli.py` | 串联 model + data + training |

**runtime.py 不负责**：optimizer 操作、数据加载、checkpoint 保存。

---

## 依赖规则

```
models/ → core/ + components/ (通过 Registry) + losses/ + metrics/
models/ ✗→ training/ （绝不可反向依赖）
```