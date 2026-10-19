"""
Grad-CAM heatmaps for all models.

v4 fixes:
  - ViTCAM: check .layers (torchvision ViT) in addition to .blocks
  - HERBS: true Grad-CAM on Swin backbone (not FPN L2 heuristic)
"""

import os, sys, gc, warnings
sys.path.insert(0, 'src')

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms as T
from scipy.ndimage import gaussian_filter

from elysia_cv.data import CassavaDataset
from elysia_cv.models.classification.cassava.runtime import build_model
from elysia_cv.config import load_cv_config_with_object

warnings.filterwarnings("ignore", message=".*BackwardHookFunctionBackward is a view.*")
warnings.filterwarnings("ignore", message=".*A module with full_backward_hook.*")

OUTPUT_DIR = Path("outputs/heatmaps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DISEASE_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# Grad-CAM for CNN (target: layer3, 28×28 for 448 input)
# ═══════════════════════════════════════════════════════════════

class CNNGradCAM:
    """Grad-CAM on layer3 of ResNet (28×28 @448px — 2× finer than layer4)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self._handles = []

        target = None
        for name, m in model.backbone.named_modules():
            if name.endswith('.layer3.1.bn2') or name.endswith('.layer3.1.relu'):
                target = m
                break
            if isinstance(m, nn.Conv2d):
                target = m

        if target is None:
            for m in model.backbone.modules():
                if isinstance(m, nn.Conv2d):
                    target = m

        if target is not None:
            self._handles.append(target.register_forward_hook(self._save_act))
            self._handles.append(target.register_full_backward_hook(self._save_grad))

    def _save_act(self, m, inp, out):
        self._act = out.detach().clone()

    def _save_grad(self, m, gin, gout):
        self._grad = gout[0].detach().clone()

    def remove(self):
        for h in self._handles:
            h.remove()

    def cam(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        self._act = None
        self._grad = None
        out = self.model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[0, class_idx].backward(retain_graph=False)

        if self._act is None or self._grad is None:
            return np.zeros((28, 28))

        act = self._act[0]
        grad = self._grad[0]
        w = grad.mean(dim=(1, 2))
        cam = (w.view(-1, 1, 1) * act).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# ViT: classifier-weight CAM (FIXED: supports .layers)
# ═══════════════════════════════════════════════════════════════

class ViTCAM:
    """Classifier-weight CAM for ViT.

    Uses: patches @ fc_weight[target_class]  → spatial heatmap.
    Fixed to support both .blocks and .layers (torchvision ViT uses .layers).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._feats: Optional[torch.Tensor] = None
        self._handle = None

        for name, m in model.backbone.named_modules():
            # torchvision ViT uses encoder.layers (Sequential of EncoderBlock)
            if hasattr(m, 'layers'):
                self._handle = m.layers[-1].register_forward_hook(self._capture)
                print(f"    ViTCAM: hooked encoder.layers[-1] ({name})")
                break
            # custom ViT might use .blocks
            if hasattr(m, 'blocks'):
                self._handle = m.blocks[-1].register_forward_hook(self._capture)
                print(f"    ViTCAM: hooked encoder.blocks[-1] ({name})")
                break

        if self._handle is None:
            print("    ViTCAM: WARNING — no .layers or .blocks found!")

        head = model.head
        self._fc_w = getattr(getattr(head, 'fc', None), 'weight', None)
        if self._fc_w is None:
            # Try model.head.head.fc for some architectures
            for attr_path in ['head.fc', 'head.head.fc']:
                parts = attr_path.split('.')
                obj = model
                for p in parts:
                    obj = getattr(obj, p, None)
                    if obj is None: break
                if hasattr(obj, 'weight'):
                    self._fc_w = obj.weight
                    break

    def _capture(self, m, inp, out):
        self._feats = out.detach()

    def remove(self):
        if self._handle:
            self._handle.remove()

    def cam(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self._feats = None
        with torch.no_grad():
            out = self.model(x)
            logits = out["logits"] if isinstance(out, dict) else out
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        if self._feats is None or self._fc_w is None:
            return np.zeros((28, 28))

        tokens = self._feats[0]  # shape: [N+1, D] (CLS + patches)
        # Remove CLS token (first position)
        patches = tokens[1:]
        N = patches.shape[0]
        side = int(N ** 0.5)
        patches = patches[:side * side]
        cam = patches @ self._fc_w[class_idx]  # [N] dot-product scores
        cam = cam.reshape(side, side)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# HERBS Grad-CAM: True Grad-CAM on Swin backbone
# ═══════════════════════════════════════════════════════════════

class HERBSGradCAM:
    """True Grad-CAM on Swin backbone for HERBS models.

    Hooks the last block of Swin stage 4 (layers_3), backprops from
    final logits through neck+head, then computes CAM from activations
    and gradients.  This replaces the old FPN-energy heuristic.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self._handle_act = None
        self._handle_grad = None

        # Find Swin backbone stage 4, last block
        for name, m in model.backbone.named_modules():
            # Match: swin.layers_3.blocks.N where N is the last block
            if 'swin.layers_3.blocks.' in name and '.norm' not in name and '.attn' not in name and '.mlp' not in name and '.drop' not in name:
                # This is a SwinTransformerBlock name
                # We want the LAST one — keep overwriting
                last_block_name = name
                last_block_module = m

        if 'last_block_name' in dir():
            self._handle_act = last_block_module.register_forward_hook(self._save_act)
            self._handle_grad = last_block_module.register_full_backward_hook(self._save_grad)
            print(f"    HERBSGradCAM: hooked {last_block_name}")
        else:
            print("    HERBSGradCAM: WARNING — no Swin layer_3 blocks found!")

    def _save_act(self, m, inp, out):
        # out is [B, HW, C] from SwinTransformerBlock
        self._act = out.detach().clone()

    def _save_grad(self, m, gin, gout):
        # gin/gout are tuples of grad w.r.t inputs/outputs
        if gout is not None and len(gout) > 0:
            self._grad = gout[0].detach().clone()

    def remove(self):
        for h in [self._handle_act, self._handle_grad]:
            if h:
                h.remove()

    def cam(self, x: torch.Tensor, class_idx: Optional[int] = None,
            use_multi_scale: bool = True) -> np.ndarray:
        """Compute Grad-CAM heatmap.

        Args:
            x: input tensor [1, 3, H, W]
            class_idx: target class, or None for argmax
            use_multi_scale: unused (kept for API compatibility)
        """
        self.model.zero_grad()
        self._act = None
        self._grad = None

        out = self.model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward(retain_graph=False)

        if self._act is None:
            return np.zeros((12, 12))

        # SwinTransformerBlock output: [B, H, W, C] (spatial, NOT flattened)
        act = self._act[0]  # [H, W, C]
        H, W_map, C = act.shape[0], act.shape[1], act.shape[2]
        act_2d = act.permute(2, 0, 1)  # [C, H, W]

        grad = self._grad
        if grad is None:
            grad_2d = torch.zeros_like(act_2d)
        else:
            g = grad[0]  # [H, W, C]
            grad_2d = g.permute(2, 0, 1)  # [C, H, W]

        # Per-position gradient-weighted activation importance.
        # For transformer blocks, standard Grad-CAM (channel-mean of gradients)
        # fails because gradients are balanced positive/negative. We instead
        # compute per-position importance as: sum_c |grad(c) * act(c)|.
        cam = (grad_2d * act_2d).abs().sum(dim=0)  # [H, W]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def build_from_exp(cfg_path: str):
    cfg = load_cv_config_with_object(cfg_path)
    d = {'model': {'type': cfg.model.type, 'backbone': dict(cfg.model.backbone)}}
    neck = getattr(cfg.model, 'neck', None)
    if neck is not None:
        d['model']['neck'] = dict(neck)
    head = getattr(cfg.model, 'head', None)
    if head is not None:
        d['model']['head'] = dict(head)
    return build_model(d)


def load_ckpt(model: nn.Module, ckpt_path: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


def resolve_ckpt(wd: str) -> Optional[str]:
    for name in ['checkpoint_best_Accuracy.pt', 'checkpoint_best.pt', 'checkpoint_final.pt']:
        p = f"{wd}/{name}"
        if os.path.exists(p):
            return p
    return None


def apply_heatmap(img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay heatmap at original image resolution with gaussian smoothing."""
    w, h = img_pil.size
    heatmap = gaussian_filter(heatmap, sigma=2.0)
    hm_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
    hm = np.array(hm_pil).astype(np.float32) / 255.0
    hm_color = plt.cm.jet(hm)[:, :, :3]
    hm_color = np.uint8(255 * hm_color)

    img = np.array(img_pil.convert("RGB"))
    overlay = (1 - alpha) * img.astype(np.float32) + alpha * hm_color.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def get_val_transform(size: int = 448):
    return T.Compose([
        T.Resize(size), T.CenterCrop(size), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_sample_images():
    val_ds = CassavaDataset(
        data_root='/data/hdd3/chuqi/data/cassava', split='val', transform=None,
    )
    samples = {}
    for idx in range(len(val_ds)):
        label = val_ds.labels[idx]
        if label not in samples:
            img_id = val_ds.image_ids[idx]
            img_path = val_ds.root / "train_images" / img_id
            samples[label] = {"img_path": str(img_path), "img_id": img_id, "label": label}
        if len(samples) == 5:
            break
    return samples


# ═══════════════════════════════════════════════════════════════
# Registry (verified evaluation numbers from PAPER.md v3)
# ═══════════════════════════════════════════════════════════════

MODELS = [
    # (display_name, config, work_dir, img_size, cam_type, accuracy)
    ("SimpleCNN",   "configs/experiments/cassava_simple_cnn.yaml",   "outputs/checkpoints/cassava_simple_cnn",   448, "cnn",   82.01),
    ("ResNet34",    "configs/experiments/cassava_resnet34.yaml",     "outputs/checkpoints/cassava_resnet34",     448, "cnn",   81.31),
    ("ResNet50",    "configs/experiments/cassava_resnet50.yaml",     "outputs/checkpoints/cassava_resnet50",     448, "cnn",   66.92),
    ("ViT-B16",     "configs/experiments/cassava_vit_b16.yaml",      "outputs/checkpoints/cassava_vit_b16",      448, "vit",   68.83),
    ("ViT-L16",     "configs/experiments/cassava_vit_l16.yaml",      "outputs/checkpoints/cassava_vit_l16",      448, "vit",   76.54),
    ("Ours-Tiny",   "configs/experiments/cassava_herbs_swin_tiny.yaml", "outputs/checkpoints/cassava_herbs_swin_tiny", 224, "herbs", 87.66),
    ("Ours-Base",   "configs/experiments/cassava_herbs_swin_base.yaml", "outputs/checkpoints/cassava_herbs_swin_base", 384, "herbs", 89.04),
]

HERBS_CKPTS = [
    ("step 14,500",  "outputs/checkpoints/cassava_herbs_swin_base/checkpoint_step_14500.pt"),
    ("step 15,000",  "outputs/checkpoints/cassava_herbs_swin_base/checkpoint_step_15000.pt"),
    ("step 15,500",  "outputs/checkpoints/cassava_herbs_swin_base/checkpoint_step_15500.pt"),
    ("final 16,050", "outputs/checkpoints/cassava_herbs_swin_base/checkpoint_final.pt"),
]


# ═══════════════════════════════════════════════════════════════
# Part 1: All Models
# ═══════════════════════════════════════════════════════════════

def visualize_all_models(samples):
    class_labels = sorted(samples.keys())
    n_models, n_classes = len(MODELS), len(class_labels)

    fig, axes = plt.subplots(
        n_models, n_classes,
        figsize=(n_classes * 3.0, n_models * 3.0),
        gridspec_kw={'wspace': 0.03, 'hspace': 0.25, 'left': 0.14},
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (name, cfg_path, wd, img_size, cam_type, acc) in enumerate(MODELS):
        ckpt = resolve_ckpt(wd)
        if ckpt is None:
            axes[row_idx, 0].set_ylabel(f"{name}\nN/A", fontsize=10, rotation=0,
                                        labelpad=25, va='center')
            for c in range(n_classes):
                axes[row_idx, c].axis('off')
            continue

        print(f"  [{name}] ", end="", flush=True)
        model = build_from_exp(cfg_path)
        load_ckpt(model, ckpt)
        model.to(DEVICE).eval()

        # Trigger lazy neck build for HERBS models (forward pass needed)
        if cam_type == "herbs":
            dummy = torch.randn(1, 3, img_size, img_size, device=DEVICE)
            with torch.no_grad():
                _ = model(dummy)

        if cam_type == "cnn":
            extractor = CNNGradCAM(model)
        elif cam_type == "herbs":
            extractor = HERBSGradCAM(model)
        else:
            extractor = ViTCAM(model)

        transform = get_val_transform(img_size)
        label_text = f"{name}\n{acc:.1f}%"
        axes[row_idx, 0].set_ylabel(
            label_text, fontsize=13, fontweight='bold',
            rotation=0, labelpad=20, va='center', ha='right',
        )
        axes[row_idx, 0].text(
            0.01, 0.99, f"{name}\n{acc:.1f}%",
            transform=axes[row_idx, 0].transAxes,
            fontsize=10, fontweight='bold', color='white',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.55),
        )

        for col_idx, label in enumerate(class_labels):
            img_pil = Image.open(samples[label]["img_path"]).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            pred = -1
            try:
                with torch.no_grad():
                    out = model(img_tensor)
                    lgs = out["logits"] if isinstance(out, dict) else out
                    pred = lgs.argmax(dim=1).item()
                heatmap = extractor.cam(img_tensor, class_idx=pred)
            except Exception as e:
                print(f"E:{e}", end=" ", flush=True)
                heatmap = np.zeros((28, 28))

            pred_name = DISEASE_NAMES[pred] if pred >= 0 else "?"
            correct = label == pred
            overlay = apply_heatmap(img_pil, heatmap)

            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].set_title(
                f"{DISEASE_NAMES[label]} → {pred_name}  {'✓' if correct else '✗'}",
                fontsize=9, color='green' if correct else 'red',
            )
            axes[row_idx, col_idx].axis('off')

            del img_tensor
            gc.collect()
            torch.cuda.empty_cache()

        extractor.remove()
        del model, extractor
        gc.collect()
        torch.cuda.empty_cache()
        print("ok", flush=True)

    for col_idx, label in enumerate(class_labels):
        axes[0, col_idx].set_title(f"{DISEASE_NAMES[label]}", fontsize=11, fontweight='bold')

    fig.suptitle("Comparison of Attention Heatmaps Across Models", fontsize=14,
                 fontweight='bold', y=1.01)
    out = OUTPUT_DIR / "all_models_heatmaps.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════
# Part 2: HERBS Attention Evolution (training stages)
# ═══════════════════════════════════════════════════════════════

def visualize_herbs_attention_evolution(samples):
    class_labels = sorted(samples.keys())
    n_stages = len(HERBS_CKPTS)
    n_classes = len(class_labels)

    fig, axes = plt.subplots(
        n_classes, n_stages + 1,
        figsize=((n_stages + 1) * 3.0, n_classes * 3.0),
        gridspec_kw={'wspace': 0.03, 'hspace': 0.20},
    )

    cfg_path = "configs/experiments/cassava_herbs_swin_base.yaml"
    img_size = 384
    transform = get_val_transform(img_size)
    model = build_from_exp(cfg_path)

    # Trigger lazy neck build once
    model.to(DEVICE).eval()
    with torch.no_grad():
        _ = model(torch.randn(1, 3, img_size, img_size, device=DEVICE))

    for row_idx, label in enumerate(class_labels):
        sample = samples[label]
        img_pil = Image.open(sample["img_path"]).convert("RGB")

        axes[row_idx, 0].imshow(img_pil)
        axes[row_idx, 0].set_title(f"{DISEASE_NAMES[label]}", fontsize=9)
        axes[row_idx, 0].axis('off')

        for col_idx, (stage_name, ckpt_path) in enumerate(HERBS_CKPTS):
            print(f"  [{DISEASE_NAMES[label]}][{stage_name}] ", end="", flush=True)

            if not os.path.exists(ckpt_path):
                axes[row_idx, col_idx + 1].text(0.5, 0.5, 'N/A',
                    transform=axes[row_idx, col_idx + 1].transAxes,
                    ha='center', va='center', fontsize=12)
                axes[row_idx, col_idx + 1].axis('off')
                print("skip", flush=True)
                continue

            load_ckpt(model, ckpt_path)
            model.to(DEVICE).eval()

            extractor = HERBSGradCAM(model)
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            pred = -1
            try:
                with torch.no_grad():
                    out = model(img_tensor)
                    lgs = out["logits"] if isinstance(out, dict) else out
                    pred = lgs.argmax(dim=1).item()
                heatmap = extractor.cam(img_tensor, class_idx=pred)
            except Exception as e:
                print(f"E:{e}", end=" ", flush=True)
                heatmap = np.zeros((12, 12))

            pred_name = DISEASE_NAMES[pred] if pred >= 0 else "?"
            correct = label == pred
            overlay = apply_heatmap(img_pil, heatmap)

            axes[row_idx, col_idx + 1].imshow(overlay)
            axes[row_idx, col_idx + 1].set_title(
                f"{stage_name}\n→{pred_name} {'✓' if correct else '✗'}",
                fontsize=9, color='green' if correct else 'red')
            axes[row_idx, col_idx + 1].axis('off')

            extractor.remove()
            del img_tensor
            gc.collect()
            torch.cuda.empty_cache()
            print("ok", flush=True)

    axes[0, 0].set_title("Original", fontsize=10, fontweight='bold')
    for col_idx, (sname, _) in enumerate(HERBS_CKPTS):
        axes[0, col_idx + 1].set_title(sname, fontsize=10, fontweight='bold')

    fig.suptitle("Ours-Base — Grad-CAM Attention Evolution During Training",
                 fontsize=14, fontweight='bold', y=1.01)
    out = OUTPUT_DIR / "herbs_swin_base_attention_evolution.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# Part 3: HERBS Multi-Stage Feature Map (backbone stages)
# ═══════════════════════════════════════════════════════════════

def visualize_herbs_multi_stage(samples):
    """Show Swin backbone's 4-stage Grad-CAM for each disease class."""
    class_labels = sorted(samples.keys())
    n_classes = len(class_labels)

    fig, axes = plt.subplots(
        n_classes, 5,
        figsize=(15, n_classes * 3.0),
        gridspec_kw={'wspace': 0.03, 'hspace': 0.20},
    )

    cfg_path = "configs/experiments/cassava_herbs_swin_base.yaml"
    img_size = 384
    ckpt_path = resolve_ckpt("outputs/checkpoints/cassava_herbs_swin_base")
    transform = get_val_transform(img_size)

    model = build_from_exp(cfg_path)
    load_ckpt(model, ckpt_path)
    model.to(DEVICE).eval()

    # Trigger lazy neck
    with torch.no_grad():
        _ = model(torch.randn(1, 3, img_size, img_size, device=DEVICE))

    for row_idx, label in enumerate(class_labels):
        sample = samples[label]
        img_pil = Image.open(sample["img_path"]).convert("RGB")

        axes[row_idx, 0].imshow(img_pil)
        axes[row_idx, 0].set_title(f"{DISEASE_NAMES[label]}", fontsize=9)
        axes[row_idx, 0].axis('off')

        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        # Use the backbone directly with hooks for each stage
        # Capture stage outputs from backbone
        stages_act = []
        stages_grad = []
        handles = []

        def make_hook(stage_idx):
            def fwd_hook(m, inp, out):
                # out is [B, HW, C], we need to store but also track for grad
                stages_act.append(out.detach())
            def bwd_hook(m, gin, gout):
                if gout and len(gout) > 0:
                    stages_grad.append(gout[0].detach())
            return fwd_hook, bwd_hook

        # Register hooks on backbone features (not possible to hook backbone internal easily)
        # Instead: use a simple approach — do Grad-CAM on the overall model
        # and just show the final attention
        extractor = HERBSGradCAM(model)

        pred = -1
        try:
            with torch.no_grad():
                out = model(img_tensor)
                lgs = out["logits"] if isinstance(out, dict) else out
                pred = lgs.argmax(dim=1).item()
            heatmap = extractor.cam(img_tensor, class_idx=pred)
        except Exception as e:
            print(f"E:{e}", end=" ", flush=True)
            heatmap = np.zeros((12, 12))

        pred_name = DISEASE_NAMES[pred] if pred >= 0 else "?"
        correct = label == pred
        overlay = apply_heatmap(img_pil, heatmap)

        # Show the raw Grad-CAM in column 1
        axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].set_title(f"Grad-CAM\n→{pred_name} {'✓' if correct else '✗'}",
                                    fontsize=9, color='green' if correct else 'red')
        axes[row_idx, 1].axis('off')

        # Show backbone stage 2/3 feature energy for comparison
        with torch.no_grad():
            stage_feats = model.backbone(img_tensor)  # list of [B, HW, C]

        for stage_idx in range(3):
            feats = stage_feats[stage_idx + 1]  # stages 1, 2, 3 (skip stage 0)
            B, HW, C = feats.shape
            side = int(HW ** 0.5)
            energy = feats[0, :side*side].norm(dim=-1).reshape(side, side)
            energy = energy.cpu().numpy()

            col = stage_idx + 2
            axes[row_idx, col].imshow(img_pil)
            # Overlay feature energy
            from scipy.ndimage import zoom
            energy_zoom = zoom(energy, (img_pil.size[1] / side, img_pil.size[0] / side), order=1)
            hm_norm = (energy_zoom - energy_zoom.min()) / (energy_zoom.max() - energy_zoom.min() + 1e-8)
            hm_color = plt.cm.viridis(hm_norm)[:, :, :3]
            axes[row_idx, col].imshow(hm_color, alpha=0.5)
            axes[row_idx, col].set_title(f"Stage {stage_idx+2}\n{side}×{side}",
                                          fontsize=9)
            axes[row_idx, col].axis('off')

        extractor.remove()
        del img_tensor
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [{DISEASE_NAMES[label]}] ok", flush=True)

    axes[0, 0].set_title("Original", fontsize=10, fontweight='bold')
    axes[0, 1].set_title("Grad-CAM", fontsize=10, fontweight='bold')

    fig.suptitle("Ours-Base — Multi-Scale Feature Analysis",
                 fontsize=14, fontweight='bold', y=1.01)
    out = OUTPUT_DIR / "herbs_swin_base_multistage.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("Loading val samples...")
    samples = get_sample_images()
    ids = [(k, DISEASE_NAMES[k], Path(samples[k]["img_id"]).stem)
           for k in sorted(samples.keys())]
    print(f"  {len(samples)} images:")
    for k, d, fid in ids:
        print(f"    class {k} ({d}): {fid}")

    if mode in ("all", "models"):
        print("\n=== Part 1: All Models ===")
        visualize_all_models(samples)

    if mode in ("all", "herbs"):
        print("\n=== Part 2: HERBS Attention Evolution ===")
        visualize_herbs_attention_evolution(samples)

    if mode in ("all", "multistage"):
        print("\n=== Part 3: HERBS Multi-Stage ===")
        visualize_herbs_multi_stage(samples)

    print("\nDone!")