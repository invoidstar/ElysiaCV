# AGENTS.md — data/

> 本模块负责数据加载：Dataset 定义、DataLoader 组装、数据增强。

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `cassava_dataset.py` | `CassavaDataset` — Cassava 病害数据集 |
| `dataset.py` | 基础 dataset 抽象 |
| `dataloader.py` | DataLoader 构建器 |
| `transforms/` | 数据增强 pipeline（augmentation、normalization、formatting） |

---

## 约定

- 所有 Dataset 类遵循 PyTorch `torch.utils.data.Dataset` 契约
- `__getitem__` 返回 dict：`{"inputs": Tensor, "gt_label": Tensor, ...}`
- 数据增强通过 `transforms/` 模块注册到 `TRANSFORMS` Registry
- DataLoader 参数（batch_size、num_workers 等）来自 YAML 配置，不在代码中硬编码

---

## 当前状态

⚠️ `data/` 模块文件已存在，但部分功能尚未完全接入 TrainerLoop。
当前 CLI 训练直接使用 `CassavaDataset` + 手动构建 `DataLoader`。

---

## 依赖规则

```
data/ → core/ + torch + torchvision
data/ ✗→ models/ / training/
```