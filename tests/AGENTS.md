# AGENTS.md — tests/

> 本目录存放 pytest 测试。所有核心功能必须有测试覆盖。

---

## 测试文件

| 文件 | 测试内容 |
|------|---------|
| `test_registry.py` | Registry 的 register / build / 重复注册 |
| `test_plugin.py` | 模型插件发现和加载 |
| `test_config.py` | 配置加载、`_base_` 继承、override 合并、schema 校验 |
| `test_cassava.py` | Cassava 模型构建 + 前向传播 |
| `test_cassava_experiments.py` | 7 个模型的完整 10-step 训练 + 配置加载 |
| `test_training.py` | TrainerLoop、HookBus、DDP |
| `test_det_seg.py` | 检测/分割模型构建 |

---

## 运行测试

```bash
# 全部测试（需要 GPU）
CUDA_VISIBLE_DEVICES=6 pytest tests/ -v

# 跳过 GPU 测试
pytest tests/ -v -k "not gpu"

# 单个测试
pytest tests/test_cassava_experiments.py -v
```

---

## 添加新测试

1. 文件名：`test_<模块名>.py`
2. 测试函数：`test_<功能描述>`
3. 需要 GPU 的测试用 `@pytest.mark.gpu` 标记
4. 模型测试至少包含：构建测试 + 前向测试 + 10-step 训练测试
5. 配置测试至少包含：正确加载 + 错误输入 + 覆盖合并