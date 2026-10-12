# AGENTS.md — tools/

> 本目录存放独立的命令行工具脚本。

---

## 文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `train.py` | 训练入口（`python tools/train.py <config>`） | ✅ 活跃 |
| `analyze.py` | 训练后分析（混淆矩阵 + GradCAM） | ✅ 活跃 |
| `test.py` | 测试入口 | ⚠️ 占位 |
| `infer.py` | 推理入口 | ⚠️ 占位 |

---

## 约定

- 每个工具文件独立运行，不相互依赖
- 参数解析使用 `argparse`
- 支持 `--set key=value` 覆盖配置
- 输出路径相对于项目根目录

---

## 添加新工具

```python
#!/usr/bin/env python3
"""工具描述。

Usage:
    python tools/my_tool.py <arg1> [--options]
"""
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# ... 实现
```