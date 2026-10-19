"""
RISE (Randomized Input Sampling for Explanation) — fair model-agnostic heatmaps.

Petsiuk et al., BMVC 2019: "RISE: Randomized Input Sampling for Explanation of Black-box Models"

All models compared with the SAME attribution method → apples-to-apples comparison.
No Grad-CAM, no classifier-weight tricks — purely occlusion-based importance.
"""

import os, sys, gc, warnings, random
sys.path.insert(0, 'src')

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms as T
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from elysia_cv.data import CassavaDataset
from elysia_cv.models.classification.cassava.runtime import build_model
from elysia_cv.config import load_cv_config_with_object

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs/heatmaps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DISEASE_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization (used for occlusion fill)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ═══════════════════════════════════════════════════════════════
# RISE: Randomized Input Sampling for Explanation
# ═══════════════════════════════════════════════════════════════

class RISE:
    """Model-agnostic explanation via random occlusion.

    Generates N binary masks at grid_size × grid_size, upsamples to image
    resolution, applies masks to the input, and computes importance as the
    weighted sum of masks where weights are target-class scores per mask.

    Reference: Petsiuk et al., "RISE: Randomized Input Sampling for
               Explanation of Black-box Models", BMVC 2019.
    """

    def __init__(self, model: torch.nn.Module, input_size: int,
                 grid_size: int = 16, N: int = 3000, p: float = 0.5,
                 batch_size: int = 64,
                 normalize_fn = None):
        """
        Args:
            model: classification model (returns dict with 'logits' or raw tensor)
            input_size: input image size (square)
            grid_size: low-res mask grid (grid_size × grid_size)
            N: number of random masks
            p: probability of a grid cell being unmasked (1 = keep, 0 = occlude)
            batch_size: forward pass batch size
            normalize_fn: optional denormalization for occlusion fill computation
        """
        self.model = model
        self.input_size = input_size
        self.grid_size = grid_size
        self.N = N
        self.batch_size = batch_size
        self.normalize_fn = normalize_fn

        # Pre-generate masks: [N, 1, grid_size, grid_size]
        print(f"  RISE: generating {N} masks ({grid_size}×{grid_size})...", end=" ", flush=True)
        masks = torch.bernoulli(torch.full((N, 1, grid_size, grid_size), p))
        self.masks = masks.to(DEVICE)
        print("done")

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None
                 ) -> np.ndarray:
        """Compute RISE importance map.

        Args:
            x: input tensor [1, 3, H, W] (normalized)
            class_idx: target class index (None = use predicted class)

        Returns:
            heatmap: numpy array [H, W] of importance scores, normalized to [0, 1]
        """
        _, _, H, W = x.shape
        self.model.eval()

        # 1. Determine target class
        with torch.no_grad():
            out = self.model(x)
            logits = out["logits"] if isinstance(out, dict) else out
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # 2. Precompute occlusion fill value (image mean in denormalized space)
        #    For normalized images, fill with zero = mean color after normalization
        fill_val = torch.zeros(3, 1, 1, device=DEVICE)

        # 3. Upsample masks to image size once
        masks_upsampled = F.interpolate(
            self.masks, size=(H, W), mode='bilinear', align_corners=False
        )  # [N, 1, H, W]

        # 4. Process in batches
        scores = torch.zeros(self.N, device=DEVICE)
        img = x.expand(1, -1, -1, -1)  # [1, 3, H, W]

        for i in range(0, self.N, self.batch_size):
            end = min(i + self.batch_size, self.N)
            batch_masks = masks_upsampled[i:end]  # [B, 1, H, W]

            # Apply masks: keep original pixels where mask=1, fill where mask=0
            masked_batch = batch_masks * img + (1 - batch_masks) * fill_val
            # masked_batch: [B, 3, H, W]

            with torch.no_grad():
                out = self.model(masked_batch)
                batch_logits = out["logits"] if isinstance(out, dict) else out
                scores[i:end] = batch_logits[:, class_idx]

        # 5. RISE formula: M = Σ (mask_i × score_i) / Σ score_i
        #    score_i = model's softmax confidence for target class under mask_i
        #    Higher score → mask keeps important pixels → higher weight
        weights = F.softmax(scores, dim=0)  # [N] normalized weights, sum=1
        importance = (masks_upsampled.squeeze(1) * weights.view(-1, 1, 1)).sum(dim=0)
        importance = F.relu(importance)

        # Normalize to [0, 1]
        importance = importance - importance.min()
        if importance.max() > 0:
            importance = importance / importance.max()

        return importance.cpu().numpy()


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


def load_ckpt(model, ckpt_path: str):
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


def get_val_transform(size: int = 448):
    return T.Compose([
        T.Resize(size), T.CenterCrop(size), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def apply_heatmap(img_pil: Image.Image, heatmap: np.ndarray,
                  alpha: float = 0.45, sigma: float = 3.0) -> Image.Image:
    """Overlay heatmap at original image resolution with gaussian smoothing."""
    w, h = img_pil.size
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    hm_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
    hm = np.array(hm_pil).astype(np.float32) / 255.0
    hm_color = plt.cm.jet(hm)[:, :, :3]
    hm_color = np.uint8(255 * hm_color)
    img = np.array(img_pil.convert("RGB"))
    overlay = (1 - alpha) * img.astype(np.float32) + alpha * hm_color.astype(np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


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
# Registry
# ═══════════════════════════════════════════════════════════════

MODELS = [
    ("SimpleCNN",  "configs/experiments/cassava_simple_cnn.yaml",   "outputs/checkpoints/cassava_simple_cnn",   448, 82.01),
    ("ResNet-34",  "configs/experiments/cassava_resnet34.yaml",     "outputs/checkpoints/cassava_resnet34",     448, 81.31),
    ("ResNet-50",  "configs/experiments/cassava_resnet50.yaml",     "outputs/checkpoints/cassava_resnet50",     448, 66.92),
    ("ViT-B/16",   "configs/experiments/cassava_vit_b16.yaml",      "outputs/checkpoints/cassava_vit_b16",      448, 68.83),
    ("ViT-L/16",   "configs/experiments/cassava_vit_l16.yaml",      "outputs/checkpoints/cassava_vit_l16",      448, 76.54),
    ("Ours-Tiny",  "configs/experiments/cassava_herbs_swin_tiny.yaml", "outputs/checkpoints/cassava_herbs_swin_tiny", 224, 87.66),
    ("Ours-Base",  "configs/experiments/cassava_herbs_swin_base.yaml", "outputs/checkpoints/cassava_herbs_swin_base", 384, 89.04),
]

# ═══════════════════════════════════════════════════════════════
# RISE parameters
# ═══════════════════════════════════════════════════════════════

RISE_N = 3000         # number of masks
RISE_GRID = 14        # low-res grid
RISE_BATCH = 64       # forward batch size
RISE_P = 0.5          # probability of keeping a pixel


# ═══════════════════════════════════════════════════════════════
# Main: generate RISE heatmaps for all models
# ═══════════════════════════════════════════════════════════════

def generate_rise_all_models(samples):
    """Generate RISE heatmaps for all models on the same 5 sample images."""
    class_labels = sorted(samples.keys())
    n_models, n_classes = len(MODELS), len(class_labels)

    fig, axes = plt.subplots(
        n_models + 1, n_classes,  # +1 for original images row
        figsize=(n_classes * 3.2, (n_models + 1) * 3.2),
        gridspec_kw={'wspace': 0.04, 'hspace': 0.22, 'left': 0.13},
    )

    # ── Row 0: original images ──
    for col_idx, label in enumerate(class_labels):
        img_pil = Image.open(samples[label]["img_path"]).convert("RGB")
        axes[0, col_idx].imshow(img_pil)
        axes[0, col_idx].set_title(DISEASE_NAMES[label], fontsize=11, fontweight='bold')
        axes[0, col_idx].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=11, fontweight='bold',
                           rotation=90, labelpad=10, va='center')

    # ── Rows 1..N: model heatmaps ──
    for row_idx, (name, cfg_path, wd, img_size, acc) in enumerate(MODELS):
        ckpt = resolve_ckpt(wd)
        if ckpt is None:
            for c in range(n_classes):
                axes[row_idx + 1, c].axis('off')
            continue

        print(f"\n{'='*60}")
        print(f"[{name}] (acc={acc:.1f}%) — RISE heatmaps")
        print(f"{'='*60}")

        # Build model
        model = build_from_exp(cfg_path)
        load_ckpt(model, ckpt)
        model.to(DEVICE).eval()

        # Trigger lazy neck for HERBS
        if 'herbs' in cfg_path:
            with torch.no_grad():
                _ = model(torch.randn(1, 3, img_size, img_size, device=DEVICE))

        # Create RISE explainer
        explainer = RISE(model, input_size=img_size,
                         grid_size=RISE_GRID, N=RISE_N,
                         batch_size=RISE_BATCH)

        # Label with model name + accuracy
        axes[row_idx + 1, 0].set_ylabel(
            f"{name}\n{acc:.1f}%", fontsize=11, fontweight='bold',
            rotation=0, labelpad=18, va='center', ha='right',
        )

        transform = get_val_transform(img_size)

        for col_idx, label in enumerate(class_labels):
            img_pil = Image.open(samples[label]["img_path"]).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            # Get prediction
            with torch.no_grad():
                out = model(img_tensor)
                logits = out["logits"] if isinstance(out, dict) else out
                pred = logits.argmax(dim=1).item()

            # Compute RISE
            try:
                heatmap = explainer(img_tensor, class_idx=pred)
            except Exception as e:
                print(f"  [RISE error: {e}]", flush=True)
                heatmap = np.zeros((img_size, img_size))

            pred_name = DISEASE_NAMES[pred]
            correct = label == pred
            overlay = apply_heatmap(img_pil, heatmap)

            axes[row_idx + 1, col_idx].imshow(overlay)
            axes[row_idx + 1, col_idx].set_title(
                f"{DISEASE_NAMES[label]} → {pred_name}  {'✓' if correct else '✗'}",
                fontsize=9, color='green' if correct else 'red',
            )
            axes[row_idx + 1, col_idx].axis('off')

            del img_tensor, heatmap
            gc.collect()

        del model, explainer
        gc.collect()
        torch.cuda.empty_cache()

    # Overall title
    fig.suptitle("Model-Agnostic RISE Heatmaps — All Models Compared Fairly\n"
                 f"({RISE_N} masks, {RISE_GRID}×{RISE_GRID} grid)",
                 fontsize=14, fontweight='bold', y=1.005)

    out = OUTPUT_DIR / "all_models_rise.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading validation samples...")
    samples = get_sample_images()
    ids = [(k, DISEASE_NAMES[k], Path(samples[k]["img_id"]).stem)
           for k in sorted(samples.keys())]
    print(f"  {len(samples)} images:")
    for k, d, fid in ids:
        print(f"    class {k} ({d}): {fid}")

    print(f"\nRISE config: N={RISE_N}, grid={RISE_GRID}×{RISE_GRID}, batch={RISE_BATCH}")
    print(f"Estimated time: ~{len(MODELS) * 5 * RISE_N / RISE_BATCH * 0.05:.0f}s")

    generate_rise_all_models(samples)
    print("\nDone!")