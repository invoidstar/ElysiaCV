# AGENTS.md — data/

> 本模块负责数据加载：Dataset 定义、DataLoader 组装、数据增强。

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `dataset.py` | 基础 dataset 抽象 |
| `dataloader.py` | DataLoader 构建器 |
| `video_dataset.py` | VideoDataset — 动作识别（Kinetics/UCF-101/CSV） |
| `det_dataset.py` | COCO/Cityscapes 检测数据集 |
| `cassava_dataset.py` | CassavaDataset — Cassava 病害数据集 |
| `imagefolder_dataset.py` | ImageFolder 通用分类数据集 |
| `transforms.py` | 视频数据增强 pipeline（10 种 transform） |

---

## 约定

- 所有 Dataset 类遵循 PyTorch `torch.utils.data.Dataset` 契约
- `__getitem__` 返回 dict：`{"inputs": Tensor, "gt_label": Tensor, ...}`
- 数据增强通过 `transforms/` 模块注册到 `TRANSFORMS` Registry
- DataLoader 参数（batch_size、num_workers 等）来自 YAML 配置，不在代码中硬编码

---

## 当前状态

✅ `data/` 模块已完全接入 TrainerLoop，支持多种数据集格式：
- 动作识别：Kinetics、UCF-101、JSON、CSV（通过 `VideoDataset`）
- 检测：COCO（通过 `COCODataset`）
- 分割：ADE20K、Cityscapes（通过 `ADE20KDataset`/`CityscapesDataset`）
- 分类：ImageFolder、Cassava（通过 `ImageFolderDataset`/`CassavaDataset`）

---

## 依赖规则

```
data/ → core/ + torch + torchvision
data/ ✗→ models/ / training/
```