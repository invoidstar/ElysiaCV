#!/usr/bin/env python3
"""Post-training analysis: confusion matrix + GradCAM heatmaps.

Usage:
    python tools/analyze.py <config.yaml> --checkpoint <model.pt> [--num-samples 10]

Example:
    python tools/analyze.py configs/classification/cassava_resnet50.yaml \\
        --checkpoint outputs/checkpoints/cassava_resnet50/checkpoint_best.pt \\
        --num-samples 5

Output (saved to outputs/checkpoints/<model>/analysis/):
    confusion_matrix.png          — normalized confusion matrix
    per_class_report.txt           — precision/recall/F1 per class
    gradcam_sample_*.png           — GradCAM heatmap overlays
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from matplotlib import pyplot as plt

CLASS_NAMES = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)",
               "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy"]

CMAP = "YlOrRd"


# ═══════════════════════════════════════════════════════════════
#  Utils
# ═══════════════════════════════════════════════════════════════
def _setup_chinese_font() -> None:
    """Try to set a CJK-aware font for matplotlib."""
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
#  Confusion Matrix
# ═══════════════════════════════════════════════════════════════
def compute_confusion_matrix(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
) -> np.ndarray:
    """Run full validation inference, return raw confusion matrix."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["inputs"].to(device, non_blocking=True)
            labels = batch["gt_label"].to(device, non_blocking=True)

            logits = model(images)["logits"]
            preds = logits.argmax(dim=1)

            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t][p] += 1

    model.train()
    return cm


def plot_confusion_matrix(cm: np.ndarray, save_path: str, class_names: list[str]) -> None:
    """Plot normalized confusion matrix and save to disk."""
    _setup_chinese_font()

    # Normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid /0
    cm_norm = cm.astype(np.float32) / row_sums

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap=CMAP, vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix (normalized)", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def per_class_report(cm: np.ndarray, class_names: list[str]) -> str:
    """Generate precision/recall/F1 per class from confusion matrix."""
    report_lines = [
        f"{'Class':<45s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'Support':>8s}",
        "-" * 75,
    ]
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        support = int(cm[i].sum())
        pred_pos = int(cm[:, i].sum())
        prec = tp / max(pred_pos, 1)
        rec = tp / max(support, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        report_lines.append(f"{name:<45s} {prec:6.4f} {rec:6.4f} {f1:6.4f} {support:8d}")

    # Overall accuracy
    total_correct = int(np.trace(cm))
    total_samples = int(cm.sum())
    acc = total_correct / max(total_samples, 1)
    report_lines.append("-" * 75)
    report_lines.append(f"{'Overall Accuracy':<45s} {'':>6s} {'':>6s} {'':>6s} {total_samples:8d}")
    report_lines.append(f"  {acc:.4f} ({total_correct}/{total_samples})")
    return "\n".join(report_lines)


# ═══════════════════════════════════════════════════════════════
#  GradCAM
# ═══════════════════════════════════════════════════════════════
class GradCAM:
    """GradCAM for CNNs (hooks into last conv layer).

    Falls back gracefully if no conv layer is found (e.g. ViT MLP).
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._target = None
        self._hook_handles = []
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._find_and_hook()

    def _find_and_hook(self) -> None:
        """Find the last Conv2d layer in the backbone and register hooks."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self._target = name

        if self._target is None:
            return  # No conv layer — skip GradCAM

        def _save_act(_, _in, out):  # noqa: ARG001
            self._activations = out

        def _save_grad(_, _in, _out):  # noqa: ARG001
            self._gradients = _out[0]

        target_module = dict(self.model.named_modules())[self._target]
        self._hook_handles.append(target_module.register_forward_hook(_save_act))
        self._hook_handles.append(target_module.register_full_backward_hook(_save_grad))

    def available(self) -> bool:
        return self._target is not None

    def __call__(self, images: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        """Return GradCAM heatmap ``(B, H, W)``. If class_idx is None, use top-1 pred."""
        if not self.available():
            raise RuntimeError("No Conv2d layer found — GradCAM unavailable.")

        self.model.zero_grad()
        outputs = self.model(images)
        logits = outputs["logits"]

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0].item())

        one_hot = torch.zeros_like(logits)
        one_hot[:, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=False)

        assert self._gradients is not None and self._activations is not None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (B, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=images.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)  # (B, H, W)

        # Normalize per image
        for b in range(cam.size(0)):
            c_min, c_max = cam[b].min(), cam[b].max()
            if c_max > c_min:
                cam[b] = (cam[b] - c_min) / (c_max - c_min)

        self._remove_hooks()
        return cam

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for display."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


def generate_gradcam_samples(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 5,
    class_names: list[str] | None = None,
) -> None:
    """Generate GradCAM heatmaps on a few val samples."""
    os.makedirs(save_dir, exist_ok=True)

    try:
        gradcam = GradCAM(model)
    except Exception:
        print("  GradCAM not available for this model (no conv layers).")
        return

    if not gradcam.available():
        print("  GradCAM not available for this model (no conv layers).")
        return

    _setup_chinese_font()
    generated = 0

    for batch in dataloader:
        if generated >= num_samples:
            break

        images = batch["inputs"].to(device, non_blocking=True)
        labels = batch["gt_label"].to(device, non_blocking=True)

        with torch.enable_grad():
            heatmaps = gradcam(images, class_idx=None)

        for b in range(min(images.size(0), num_samples - generated)):
            img = denormalize(images[b].cpu())
            hm = heatmaps[b].cpu().numpy()
            true_label = int(labels[b].item())
            true_name = class_names[true_label] if class_names else str(true_label)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            axes[0].imshow(img)
            axes[0].set_title(f"Original\nTrue: {true_name}", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(img)
            axes[1].imshow(hm, cmap="jet", alpha=0.5)
            axes[1].set_title("GradCAM Overlay", fontsize=10)
            axes[1].axis("off")

            save_path = os.path.join(save_dir, f"gradcam_sample_{generated + 1:02d}.png")
            plt.tight_layout()
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            generated += 1

    print(f"  GradCAM samples saved → {save_dir}/ (×{generated})")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training analysis: confusion matrix + GradCAM")
    parser.add_argument("config", help="path to YAML config file")
    parser.add_argument("--checkpoint", "-c", required=True, help="path to trained model checkpoint")
    parser.add_argument("--num-samples", type=int, default=5, help="number of GradCAM samples (default: 5)")
    parser.add_argument("--no-gradcam", action="store_true", help="skip GradCAM (confusion matrix only)")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from elysia_cv.config import load_cv_config_with_object
    from elysia_cv.models.classification.cassava.runtime import build_model
    from elysia_cv.data import CassavaDataset

    # Load config
    print(f"Loading config: {args.config}")
    cfg = load_cv_config_with_object(args.config)

    # Determine output dir
    output_dir = cfg.run.output_dir
    output_dir = os.path.abspath(output_dir)
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Analysis dir: {analysis_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model + load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model_cfg = {
        "model": {
            "type": cfg.model.type,
            "backbone": dict(cfg.model.backbone),
            "head": dict(cfg.model.head),
        }
    }
    model = build_model(model_cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    model = model.to(device)
    model.eval()

    # Dataloader
    bb_type = dict(cfg.model.backbone).get("type", "ResNet")
    img_size = 64 if bb_type == "MLP" else 448
    val_transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_ds = CassavaDataset(cfg.data.val.dataset_root, split="val", transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # ═════════════════════════════════════════════════════════
    #  1. Confusion Matrix
    # ═════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("  1. Confusion Matrix")
    print("─" * 50)
    cm = compute_confusion_matrix(model, val_loader, device, num_classes=5)
    cm_path = os.path.join(analysis_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path, CLASS_NAMES)

    report = per_class_report(cm, CLASS_NAMES)
    report_path = os.path.join(analysis_dir, "per_class_report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"  Per-class report saved → {report_path}")
    print()
    print(report)

    # ═════════════════════════════════════════════════════════
    #  2. GradCAM
    # ═════════════════════════════════════════════════════════
    if not args.no_gradcam:
        print("\n" + "─" * 50)
        print("  2. GradCAM Heatmaps")
        print("─" * 50)
        gradcam_dir = os.path.join(analysis_dir, "gradcam")
        generate_gradcam_samples(model, val_loader, device, gradcam_dir,
                                 num_samples=args.num_samples, class_names=CLASS_NAMES)

    print("\n" + "=" * 50)
    print(f"  Analysis complete! → {analysis_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()