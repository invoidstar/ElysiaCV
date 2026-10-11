"""ResNet backbone — registered via BACKBONES registry.

Wraps torchvision's ResNet, stripping the classification head.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from elysia_cv.core.registry import BACKBONES


@BACKBONES.register_module()
class ResNet(nn.Module):
    """ResNet backbone from torchvision.

    Args:
        depth: ResNet depth variant (18, 34, 50, 101, 152).
        pretrained: Load torchvision pretrained weights.
        frozen_stages: Freeze the first N stages (-1 = none).
        out_indices: Which stage outputs to return (0-based).
        pool_last: If True and single output, avgpool + flatten (classification).
            Set to False for detection/segmentation which need spatial features.
    """

    _DEPTH_TO_ARCH = {
        18: "resnet18",
        34: "resnet34",
        50: "resnet50",
        101: "resnet101",
        152: "resnet152",
    }

    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = False,
        frozen_stages: int = -1,
        out_indices: tuple[int, ...] | None = None,
        pool_last: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        import torchvision.models as tv_models

        arch = self._DEPTH_TO_ARCH.get(depth)
        if arch is None:
            raise ValueError(f"unsupported ResNet depth: {depth}")

        weights = "DEFAULT" if pretrained else None
        full_model = getattr(tv_models, arch)(weights=weights)

        # Strip head — keep backbone + avgpool
        self.stem = nn.Sequential(
            full_model.conv1,
            full_model.bn1,
            full_model.relu,
            full_model.maxpool,
        )
        self.layer1 = full_model.layer1
        self.layer2 = full_model.layer2
        self.layer3 = full_model.layer3
        self.layer4 = full_model.layer4
        self.avgpool = full_model.avgpool

        self._out_indices = out_indices or (4,)  # default: final stage only
        self.pool_last = pool_last
        self._out_channels = self._compute_out_channels()

        # Freeze early stages
        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

    def _compute_out_channels(self) -> list[int]:
        """Compute output channels per stage for FPN/Neck compatibility."""
        channels = []
        x = None
        for stage_fn in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            if x is None:
                x = stage_fn.__class__  # placeholder
                continue
        # Hard-coded for ResNet family
        return {
            18: [64, 64, 128, 256, 512],
            34: [64, 64, 128, 256, 512],
            50: [64, 256, 512, 1024, 2048],
            101: [64, 256, 512, 1024, 2048],
            152: [64, 256, 512, 1024, 2048],
        }.get(getattr(self, "_depth", 50), [64, 256, 512, 1024, 2048])

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    def _freeze_stages(self, num_stages: int) -> None:
        """Freeze batch-norm and parameters of early stages."""
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for idx in range(min(num_stages, len(stages))):
            stage = stages[idx]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x: Any) -> Any:
        """Return outputs at ``self._out_indices``.

        If only one index, returns a single tensor; otherwise a tuple.
        """
        outputs: list[Any] = []
        x = self.stem(x)
        if 0 in self._out_indices:
            outputs.append(x)
        x = self.layer1(x)
        if 1 in self._out_indices:
            outputs.append(x)
        x = self.layer2(x)
        if 2 in self._out_indices:
            outputs.append(x)
        x = self.layer3(x)
        if 3 in self._out_indices:
            outputs.append(x)
        x = self.layer4(x)
        if 4 in self._out_indices:
            outputs.append(x)

        if len(outputs) == 0:
            outputs.append(x)

        if len(outputs) == 1:
            single = outputs[0]
            if self.pool_last:
                # Classification: avgpool → flatten
                pooled = self.avgpool(single)
                return pooled.flatten(1)
            return single

        return tuple(outputs) if len(outputs) > 1 else outputs[0]