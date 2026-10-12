# AGENTS.md — core/

> 本模块是 ElysiaCV 的零依赖基础设施。修改前请理解以下约定。

---

## 职责边界

`core/` 是项目最底层，**不允许** import torch 或任何项目内模块。

```
core/ ← 不被任何模块依赖，但可以被所有模块 import
```

---

## 关键文件

| 文件 | 作用 | 修改时注意 |
|------|------|-----------|
| `registry.py` | `Registry` 类：`register_module()` + `build()` | 全局单例，不要新建 Registry 实例 |
| `plugin.py` | `PluginBase` 抽象基类（模型插件契约） | 新增任务类型需继承此基类 |
| `api_version.py` | `CORE_API_VERSION = "1.0"`，主版本兼容性检查 | 破坏性变更时升级主版本号 |
| `errors.py` | 自定义异常类 | 所有框架级异常统一从这里抛出 |
| `capabilities.py` | `Capability` 枚举 | 模型声明自己支持的能力 |
| `readiness.py` | 健康检查 | 训练前校验 |

---

## Registry 模式（重要！）

```python
# 定义 Registry（在 core/ 中）
from elysia_cv.core.registry import Registry

BACKBONES = Registry("backbones")

# 注册组件（在各 components/ 文件中）
@BACKBONES.register_module()
class ResNet(nn.Module):
    ...

# 构建组件（在运行时）
backbone = BACKBONES.build({"type": "ResNet", "depth": 50})
```

**所有 Registry 在 `core/registry.py` 顶部已预先声明**：
- `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`, `METRICS`, `TRANSFORMS`, `HOOKS`

新增 Registry 时在 `core/registry.py` 底部添加一行。

---

## 类型约定

- 所有 dict 操作使用 `typing.Any` 作为值类型
- 公开 API 必须有类型标注
- 私有函数可以省略部分标注