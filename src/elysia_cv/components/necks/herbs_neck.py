"""HERBS Neck — High-temperaturE Refinement and Background Suppression.

Adapted from PIM (Plug-in Module) in FGVC-HERBS repository:
https://github.com/chou141253/FGVC-HERBS

The neck takes multi-scale backbone features (4 levels), applies FPN fusion,
confidence-based discriminative token selection (WeaklySelector), and a
graph convolution combiner (GCNCombiner) for fine-grained classification.

The original "high-temperature refinement" corresponds to selecting top-k
tokens per layer based on classification confidence scores — focusing on
discriminative foreground regions and suppressing background.

Paper: https://arxiv.org/abs/2303.06442
"""

from __future__ import annotations

import copy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from elysia_cv.core.registry import NECKS


# ============================================================================
# GCN Combiner — graph convolution on selected discriminative tokens
# ============================================================================

class GCNCombiner(nn.Module):
    """Graph convolution network that combines multi-layer selected tokens.

    Parameters
    ----------
    total_num_selects : int
        Total number of tokens selected across all layers.
    num_classes : int
        Number of output classes.
    in_dim : int
        Input feature dimension per token.
    """

    def __init__(self, total_num_selects: int, num_classes: int, in_dim: int) -> None:
        super().__init__()
        self.proj_size = in_dim

        num_joints = total_num_selects // 64
        self.param_pool0 = nn.Linear(total_num_selects, num_joints)

        A = torch.eye(num_joints) / 100 + 1 / 100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(in_dim, in_dim, 1)
        self.batch_norm1 = nn.BatchNorm1d(in_dim)

        self.conv_q1 = nn.Conv1d(in_dim, in_dim // 4, 1)
        self.conv_k1 = nn.Conv1d(in_dim, in_dim // 4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        self.param_pool1 = nn.Linear(num_joints, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(in_dim, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """x: dict of {name: [B, S, C]} selected token features."""
        hs = torch.cat(list(x.values()), dim=1).transpose(1, 2).contiguous()  # [B, C, S']
        hs = self.param_pool0(hs)  # [B, C, J]

        # Adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)  # [B, J]
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1

        # Graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)

        # Classifier
        hs = self.param_pool1(hs)  # [B, C, 1]
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        return self.classifier(hs)


# ============================================================================
# WeaklySelector — confidence-based discriminative token selection
# ============================================================================

class WeaklySelector(nn.Module):
    """Select top-k tokens per layer based on classification confidence.

    This is the "high-temperature refinement" component: it uses per-layer
    classifiers to compute confidence scores, then selects the tokens with
    the highest confidence for the predicted class. Low-confidence (background)
    tokens are dropped — implementing background suppression.

    Parameters
    ----------
    feat_dims : dict
        Mapping from layer name → feature channel dimension.
    num_classes : int
        Number of output classes.
    num_selects : dict
        Mapping from layer name → number of tokens to select.
    """

    def __init__(
        self,
        feat_dims: dict[str, int],
        num_classes: int,
        num_selects: dict[str, int],
    ) -> None:
        super().__init__()
        self.num_select = num_selects
        self.num_classes = num_classes

        for name, dim in feat_dims.items():
            self.add_module(f"classifier_l_{name}", nn.Linear(dim, num_classes))

    def forward(
        self,
        x: dict[str, torch.Tensor],
        logits: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Select top-k tokens per layer based on predicted-class confidence.

        Returns dict: {name: [B, K, C]} selected token features.
        """
        selections: dict[str, torch.Tensor] = {}
        for name in x:
            if "FPN1_" in name:
                continue
            B, S, C = x[name].shape
            # Per-layer classifier (if not provided externally)
            if name not in logits:
                logits[name] = getattr(self, f"classifier_l_{name}")(x[name])

            logit = logits[name]  # [B, S, num_classes]
            probs = torch.softmax(logit, dim=-1)
            sum_probs = probs.mean(1)  # [B, num_classes]  average over spatial

            k = self.num_select.get(name, 32)
            batch_selections = []
            for bi in range(B):
                max_cls = torch.argmax(sum_probs[bi])
                _, ranks = torch.sort(probs[bi, :, max_cls], descending=True)
                batch_selections.append(x[name][bi][ranks[:k]])

            selections[name] = torch.stack(batch_selections)  # [B, K, C]

        return selections


# ============================================================================
# Multi-scale FPN for Swin Transformer (Conv1d-based, B, S, C format)
# ============================================================================

class MultiScaleFPN(nn.Module):
    """Feature Pyramid Network for sequence-format features (B, S, C).

    Unlike the standard 2D FPN, this operates on 1D token sequences
    (e.g., Swin Transformer flattened patch tokens). Uses Conv1d for
    projection and up/down-sampling fusion.

    Parameters
    ----------
    in_dims : dict
        Mapping from layer name → input channel dimension.
    fpn_size : int
        Common channel dimension for all FPN levels.
    """

    def __init__(self, in_dims: dict[str, int], fpn_size: int) -> None:
        super().__init__()
        self.fpn_size = fpn_size

        # Top-down pathway (upsample + add)
        layer_names = list(in_dims.keys())
        for i, name in enumerate(layer_names):
            # Project to common dimension
            self.add_module(
                f"proj_down_{name}",
                nn.Sequential(
                    nn.Linear(in_dims[name], in_dims[name]),
                    nn.ReLU(),
                    nn.Linear(in_dims[name], fpn_size),
                ),
            )
            # Upsample conv for fusion (except top layer)
            if i != 0:
                self.add_module(
                    f"up_{name}",
                    nn.Conv1d(fpn_size, fpn_size, 1),
                )

            # Per-level classifier (intermediate supervision)
            self.add_module(
                f"cls_down_{name}",
                nn.Sequential(
                    nn.Conv1d(fpn_size, fpn_size, 1),
                    nn.BatchNorm1d(fpn_size),
                    nn.ReLU(),
                    nn.Conv1d(fpn_size, fpn_size, 1),  # output = fpn_size for later
                ),
            )

        # Bottom-up pathway (downsample + add)
        for i, name in enumerate(layer_names):
            self.add_module(
                f"proj_up_{name}",
                nn.Sequential(
                    nn.Linear(fpn_size, fpn_size),
                    nn.ReLU(),
                    nn.Linear(fpn_size, fpn_size),
                ),
            )
            # Downsample conv for fusion (except bottom layer)
            if i != len(layer_names) - 1:
                next_name = layer_names[i + 1]
                self.add_module(
                    f"down_{name}",
                    nn.Conv1d(fpn_size, fpn_size, 1),
                )

        self._layer_names = layer_names

    def _key(self, name: str) -> str:
        """Strip 'layer' prefix if needed."""
        return name.replace("FPN1_", "")

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run FPN fusion (top-down then bottom-up).

        Args:
            x: dict {layer_name: [B, S, C]} backbone features.

        Returns:
            dict with both intermediate and fused features,
            prefixed with 'FPN1_' for down-path outputs.
        """
        names = [n for n in self._layer_names if n in x]

        # --- Top-down ---
        for name in names:
            x[name] = getattr(self, f"proj_down_{name}")(x[name])

        x[f"FPN1_{names[-1]}"] = x[names[-1]]  # top feature unchanged

        for i in range(len(names) - 1, 0, -1):
            hi_name = names[i]
            lo_name = names[i - 1]
            hi = x[hi_name].transpose(1, 2)  # [B, C, S]
            lo = x[lo_name].transpose(1, 2)
            hi_up = getattr(self, f"up_{hi_name}")(hi)
            # Match spatial sizes
            if lo.size(-1) != hi_up.size(-1):
                hi_up = F.interpolate(hi_up, size=lo.size(-1), mode="linear")
            x[lo_name] = (lo + hi_up).transpose(1, 2)
            x[f"FPN1_{lo_name}"] = x[lo_name]

        # --- Bottom-up ---
        for name in names:
            x[name] = getattr(self, f"proj_up_{name}")(x[name])

        for i in range(len(names) - 1):
            lo_name = names[i]
            hi_name = names[i + 1]
            lo = x[lo_name].transpose(1, 2)
            hi = x[hi_name].transpose(1, 2)
            lo_dn = getattr(self, f"down_{lo_name}")(lo)
            if hi.size(-1) != lo_dn.size(-1):
                lo_dn = F.interpolate(lo_dn, size=hi.size(-1), mode="linear")
            hi_fused = hi + lo_dn
            x[hi_name] = hi_fused.transpose(1, 2)

        return x


# ============================================================================
# HERBS Neck — full HERBS pipeline
# ============================================================================

@NECKS.register_module()
class HERBSNeck(nn.Module):
    """HERBS fine-grained classification neck.

    Pipeline:
    1. MultiScaleFPN: top-down + bottom-up multi-scale fusion
    2. WeaklySelector: confidence-based token selection (background suppression)
    3. GCNCombiner: graph convolution classification

    Can be used with any backbone that outputs multi-scale features
    (e.g. Swin with ``features_only=True``, ResNet with intermediate outputs).

    Parameters
    ----------
    in_dims : dict or None
        Mapping {layer_name: channels}. If None, auto-detected from input.
    num_classes : int
        Number of output classes.
    fpn_size : int
        FPN projection dimension.
    num_selects : dict or None
        Mapping {layer_name: num_tokens_to_select}. Defaults to 32 per layer.
    """

    def __init__(
        self,
        in_dims: dict[str, int] | None = None,
        num_classes: int = 5,
        fpn_size: int = 512,
        num_selects: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.num_classes = num_classes
        self.fpn_size = fpn_size
        self.num_selects = num_selects or {}
        self._built = in_dims is not None

        if in_dims is not None:
            self._build(in_dims)

    def _build(self, in_dims: dict[str, int]) -> None:
        layer_names = sorted(in_dims.keys())
        # Default: 32 tokens per layer
        default_selects = {n: self.num_selects.get(n, 32) for n in layer_names}

        self.fpn = MultiScaleFPN(in_dims, self.fpn_size)
        self.selector = WeaklySelector(
            {n: self.fpn_size for n in layer_names},
            self.num_classes,
            default_selects,
        )
        self.combiner = GCNCombiner(
            sum(default_selects.values()),
            self.num_classes,
            self.fpn_size,
        )
        self._layer_names = layer_names
        self._built = True

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            feats: List of backbone feature maps.
                   Each should be [B, S, C] for sequence features
                   or [B, C, H, W] for 2D features.

        Returns:
            Logits tensor [B, num_classes].
        """
        # Convert list to dict if needed
        if isinstance(feats, (list, tuple)):
            # Auto-name: layer1, layer2, ...
            if not self._built:
                in_dims = {}
                for i, f in enumerate(feats):
                    name = f"layer{i + 1}"
                    # Determine channel dim
                    if len(f.shape) == 4:  # [B, C, H, W]
                        in_dims[name] = f.shape[1]
                    elif len(f.shape) == 3:  # [B, S, C]
                        in_dims[name] = f.shape[-1]
                    else:
                        in_dims[name] = f.shape[1]
                # Build modules on the correct device
                device = feats[0].device
                self._build(in_dims)
                self.to(device)

            x = {f"layer{i + 1}": f for i, f in enumerate(feats)}
        else:
            x = dict(feats)
            if not self._built:
                in_dims = {}
                sample_f = next(iter(x.values()))
                for name, f in x.items():
                    if len(f.shape) == 4:
                        in_dims[name] = f.shape[1]
                    elif len(f.shape) == 3:
                        in_dims[name] = f.shape[-1]
                    else:
                        in_dims[name] = f.shape[1]
                self._build(in_dims)
                self.to(sample_f.device)

        # Flatten 2D features to sequences [B, C, H, W] → [B, H*W, C]
        for name in list(x.keys()):
            if len(x[name].shape) == 4:
                B, C, H, W = x[name].shape
                x[name] = x[name].view(B, C, H * W).permute(0, 2, 1).contiguous()

        # 1. FPN fusion
        x = self.fpn(x)

        # 2. Token selection
        logits: dict[str, torch.Tensor] = {}
        selections = self.selector(x, logits)

        # 3. Graph classification
        return self.combiner(selections)