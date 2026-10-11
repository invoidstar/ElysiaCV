# Tools

> 状态：Active（长期有效） | 最后更新：2026-10

ElysiaCV 的统一 CLI 入口。

---

## 命令

| 命令 | 用途 | 用法 |
|------|------|------|
| `train.py` | 启动训练 | `python tools/train.py config.yaml [--set k=v] [--resume ckpt.pt]` |
| `test.py` | 模型评测 | `python tools/test.py config.yaml checkpoint.pt [--out result.json]` |
| `infer.py` | 单张推理 | `python tools/infer.py config.yaml checkpoint.pt -i img.jpg [-o result.json]` |

---

## 参数说明

### train.py

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | 位置参数 | 配置文件路径 |
| `--set` | 可选，多值 | 覆盖配置值，格式 `key=value` |
| `--resume` | 可选 | 从 checkpoint 恢复训练 |
| `--no-compile` | 可选 | 关闭 torch.compile |
| `--test-run` | 可选 | 只跑 10 步（快速验证） |

### test.py

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | 位置参数 | 配置文件路径 |
| `checkpoint` | 位置参数 | 模型权重 `.pt` 路径 |
| `--set` | 可选 | 覆盖配置值 |
| `--out` | 可选 | 评测结果输出 JSON 路径 |

### infer.py

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | 位置参数 | 配置文件路径 |
| `checkpoint` | 位置参数 | 模型权重 `.pt` 路径 |
| `-i / --input` | 必须 | 输入图片路径 |
| `-o / --output` | 可选 | 推理结果输出 JSON 路径 |
| `--show` | 可选 | 显示图片（需要 GUI） |

---

## 示例

```bash
# 训练：覆盖 batch size 和 学习率
python tools/train.py configs/classification/resnet50_imagenet.yaml \
    --set data.train.batch_size=128 optim.params.lr=0.05

# 训练：dry-run 验证
python tools/train.py configs/detection/faster_rcnn_resnet50_coco.yaml \
    --test-run

# 训练：恢复
python tools/train.py configs/classification/resnet50_imagenet.yaml \
    --resume ./work_dirs/checkpoint_step_5000.pt

# 评测
python tools/test.py configs/classification/resnet50_imagenet.yaml \
    ./work_dirs/checkpoint_final.pt --out metrics.json

# 推理
python tools/infer.py configs/classification/resnet50_imagenet.yaml \
    ./work_dirs/checkpoint_best_Accuracy.pt -i photo.jpg
```