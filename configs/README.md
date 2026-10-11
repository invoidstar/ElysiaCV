# 配置索引

> 状态：Active（随配置变更更新） | 最后更新：2026-10

---

## 配置目录

```
configs/
├── README.md
├── _base_/
│   └── default_runtime.yaml          # 所有配置的基础模板
├── classification/
│   └── resnet50_imagenet.yaml        # ResNet-50 ImageNet 分类
├── detection/
│   └── faster_rcnn_resnet50_coco.yaml # Faster R-CNN COCO 检测
└── segmentation/
    └── fcn_resnet50_ade20k.yaml      # FCN ADE20K 分割
```

---

## `_base_/default_runtime.yaml`

所有配置通过 `_base_: "../_base_/default_runtime.yaml"` 继承的公共默认值：

| 节 | 内容 |
|----|------|
| `engine.distributed` | DDP 默认参数（bucket_cap_mb=64, gradient_as_bucket_view=true） |
| `engine.precision` | CUDA 后端配置（cudnn.benchmark=true, allow_tf32=true） |
| `engine.compile` | torch.compile 默认关闭 |
| `engine.optimization` | gradient_accumulation_steps=1 |
| `tracking` | log_freq=50, wandb 关闭 |
| `run` | seed=42, test_run=false |
| `eval` | eval_freq=5000, save_final=true |

---

## 使用方式

```bash
# 直接加载
python tools/train.py configs/classification/resnet50_imagenet.yaml

# 覆盖数据集路径
python tools/train.py configs/detection/faster_rcnn_resnet50_coco.yaml \
    --set data.train.dataset_root=/home/user/coco/images/train2017

# 覆盖训练参数
python tools/train.py configs/classification/resnet50_imagenet.yaml \
    --set schedule.steps=50000 optim.params.lr=0.05
```

---

## 新增配置

在对应任务目录下创建 `.yaml` 文件：

```yaml
_base_: "../_base_/default_runtime.yaml"

run:
  name: my_experiment
  output_dir: ./work_dirs/my_experiment

model:
  type: my_model
  backbone:
    type: ResNet
    depth: 50
  head:
    type: LinearClsHead
    num_classes: 10

data:
  train:
    dataset_root: /path/to/data
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
  steps: 50000
  base_lr: 0.01

eval:
  eval_freq: 5000
  metrics:
    - Accuracy
```