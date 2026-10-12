# AGENTS.md — config/

> 本模块负责 YAML 配置加载、`_base_` 继承、strict key 校验。

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `loader.py` | `load_cv_config()` — YAML → dict，含 `_base_` 解析与环境变量展开 |
| `config.py` | `CVConfig` dataclass — dict → 类型化对象 |
| `sections.py` | `DataSection`、`EngineSection` 等 dataclass 定义 |
| `validation.py` | 配置 schema 校验 |

---

## 配置加载流水线

```
User YAML → _load_yaml_with_base() → expand_environment_values() → CVConfig.from_mapping()
              ↑ _base_ 递归合并       ↑ ${VAR} 展开
```

---

## `_base_` 合并逻辑（重要！）

在 `loader.py` 的 `_load_yaml_with_base()` 中：

1. 读取当前 YAML，提取 `_base_` 段
2. 对每个 `_base_` 引用路径（相对于当前文件目录），递归加载
3. 所有 base config 用 `_deep_merge` 从左到右合并
4. 最后把当前 YAML 的内容合并到最上层（子节点覆盖父节点）

```python
# loader.py 核心逻辑
merged = {}
for ref in base_refs:          # _base_: [a.yaml, b.yaml]
    merged = _deep_merge(merged, _load_yaml_with_base(ref_path))
merged = _deep_merge(merged, loaded)  # 当前文件覆盖
```

---

## strict key 校验

`_CV_FIELD_MAP` 定义了所有合法 key 的白名单。`_OPEN_PREFIXES` 中的路径允许任意子 key：

```python
_OPEN_PREFIXES = (
    ("model", "backbone"),  # backbone 参数随类型变化，不限制
    ("model", "head"),
    ("model", "kwargs"),
    ("optim", "params"),
    ("schedule", "params"),
)
```

**新增合法 key**：在 `_CV_FIELD_MAP` 和 `_CV_TOP_LEVEL_SECTIONS` 中添加。

---

## 依赖规则

```
config/ → core/ + PyYAML + draccus
config/ ✗→ torch / models / training
```