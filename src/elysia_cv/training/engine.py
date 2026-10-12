"""TrainingEngine — wires config, model, data, and loop into a single entry point.

This is the equivalent of ElysiaRobot's native_engine.py. It takes a
:class:`CVConfig` and builds all components, then runs the training loop.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from elysia_cv.config.config import CVConfig
from elysia_cv.training.distributed import (
    DistributedContext,
    configure_cuda_backend,
    init_distributed_from_env,
    destroy_distributed,
    wrap_ddp,
)
from elysia_cv.training.strategy import ExecutionStrategy
from elysia_cv.training.cadence import CadenceController
from elysia_cv.training.loop import TrainerLoop
from elysia_cv.training.hooks.base import HookBus
from elysia_cv.training.hooks.logger import LoggerHook
from elysia_cv.training.hooks.checkpoint import CheckpointHook
from elysia_cv.training.hooks.lr_scheduler import LrSchedulerHook
from elysia_cv.training.hooks.evaluation import EvalHook
from elysia_cv.optim.optimizers import build_optimizer
from elysia_cv.optim.schedulers import build_scheduler


class TrainingEngine:
    """Top-level training entry point.

    Usage::

        config = load_cv_config_with_object("configs/experiments/cassava_resnet50.yaml")
        engine = TrainingEngine(
            config,
            build_model_fn=_my_build_model,
            build_train_loader_fn=_my_build_train_loader,
            build_val_loader_fn=_my_build_val_loader,
        )
        engine.run()
    """

    def __init__(
        self,
        config: CVConfig,
        *,
        build_model_fn: Callable[[], torch.nn.Module] | None = None,
        build_train_loader_fn: Callable[[], Any] | None = None,
        build_val_loader_fn: Callable[[], Any | None] | None = None,
    ) -> None:
        self.config = config
        self._dist_ctx: DistributedContext | None = None
        self._build_model_fn = build_model_fn
        self._build_train_loader_fn = build_train_loader_fn
        self._build_val_loader_fn = build_val_loader_fn

    def run(self) -> None:
        """Full training pipeline: init → build → loop → cleanup."""
        # 1. Distributed init + CUDA
        self._dist_ctx = init_distributed_from_env()
        configure_cuda_backend()
        device = torch.device(f"cuda:{self._dist_ctx.local_rank}" if torch.cuda.is_available() else "cpu")

        # 2. Build dataloaders
        train_loader = self._build_train_loader()
        val_loader = self._build_val_loader()

        # 3. Build model
        model = self._build_model()
        model = model.to(device)
        model = wrap_ddp(
            model, self._dist_ctx,
            find_unused_parameters=self.config.engine.distributed.ddp_find_unused_parameters,
            broadcast_buffers=self.config.engine.distributed.ddp_broadcast_buffers,
            bucket_cap_mb=self.config.engine.distributed.ddp_bucket_cap_mb,
            gradient_as_bucket_view=self.config.engine.distributed.ddp_gradient_as_bucket_view,
            static_graph=self.config.engine.distributed.ddp_static_graph,
        )

        # 4. Build optimizer + scheduler
        optimizer = build_optimizer(model, {"type": self.config.optim.type, **self.config.optim.params})
        scheduler = build_scheduler(optimizer, {
            "type": self.config.schedule.type,
            "steps": self.config.schedule.steps,
            "base_lr": self.config.schedule.base_lr,
            "min_lr": self.config.schedule.min_lr,
            "warmup_steps": self.config.schedule.warmup_steps,
            "params": dict(self.config.schedule.params),
        })

        # 5. Build strategy
        strategy = ExecutionStrategy(
            grad_accum_steps=self.config.engine.optimization.gradient_accumulation_steps,
            grad_clip_norm=self.config.engine.optimization.grad_clip_norm,
        )

        # 6. Build hooks
        hooks = HookBus()
        hooks.register(LoggerHook(
            log_freq=self.config.tracking.log_freq,
            total_steps=self.config.schedule.steps,
        ))
        hooks.register(LrSchedulerHook())
        hooks.register(CheckpointHook(
            output_dir=self.config.run.output_dir,
            save_freq=self.config.eval.eval_freq,  # same cadence as validation
            save_best_metric=self.config.eval.save_best,
            save_final=self.config.eval.save_final,
        ))
        if val_loader is not None:
            hooks.register(EvalHook(
                dataloader=val_loader,
                eval_freq=self.config.eval.eval_freq,
                metrics=self.config.eval.metrics,
                save_best_metric=self.config.eval.save_best,
            ))

        # 7. Build cadence
        cadence = CadenceController(
            log_freq=self.config.tracking.log_freq,
            eval_freq=self.config.eval.eval_freq,
        )

        # 8. Loop
        loop = TrainerLoop(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            max_steps=self.config.schedule.steps,
            strategy=strategy,
            callback=hooks,
            scheduler=scheduler,
            distributed=self._dist_ctx,
            cadence=cadence,
        )

        # 9. Run
        if self._dist_ctx.is_main_process:
            ddp_info = f" (DDP, world_size={self._dist_ctx.world_size})" if self._dist_ctx.enabled else ""
            print(f"  Engine: TrainerLoop{ddp_info}, device={device}", flush=True)

        try:
            loop.run()
        finally:
            destroy_distributed(self._dist_ctx)

    # ------------------------------------------------------------------
    # Build helpers — override via constructor functions or subclass
    # ------------------------------------------------------------------

    def _build_train_loader(self) -> Any:
        """Build training DataLoader from config, or use injected function."""
        if self._build_train_loader_fn is not None:
            return self._build_train_loader_fn()
        # Default: generic ImageFolder loader
        from elysia_cv.data.dataloader import build_dataloader
        from elysia_cv.data.dataset import ImageFolderDataset

        train_cfg = self.config.data.train
        dataset = ImageFolderDataset(
            root=train_cfg.dataset_root or "",
            split=train_cfg.split,
            transforms=list(train_cfg.transforms),
        )
        return build_dataloader(
            dataset,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            prefetch_factor=train_cfg.prefetch_factor,
            pin_memory=train_cfg.pin_memory,
            persistent_workers=train_cfg.persistent_workers,
            drop_last=train_cfg.drop_last,
            shuffle=True,
        )

    def _build_val_loader(self) -> Any | None:
        """Build validation DataLoader from config, or use injected function."""
        if self._build_val_loader_fn is not None:
            return self._build_val_loader_fn()
        val_cfg = self.config.data.val
        if not val_cfg.dataset_root:
            return None
        from elysia_cv.data.dataloader import build_dataloader
        from elysia_cv.data.dataset import ImageFolderDataset

        dataset = ImageFolderDataset(
            root=val_cfg.dataset_root or "",
            split=val_cfg.split,
            transforms=list(val_cfg.transforms),
        )
        return build_dataloader(
            dataset,
            batch_size=val_cfg.batch_size,
            num_workers=val_cfg.num_workers,
            shuffle=False,
        )

    def _build_model(self) -> torch.nn.Module:
        """Build model from config, or use injected function.

        Priority: injected function > plugin system > torchvision fallback.
        """
        if self._build_model_fn is not None:
            return self._build_model_fn()

        # Try plugin system (cassava runtime)
        model_type = self.config.model.type
        try:
            from elysia_cv.models.classification.cassava.runtime import build_model
            model_cfg = {
                "model": {
                    "type": model_type,
                    "backbone": dict(self.config.model.backbone),
                    "head": dict(self.config.model.head),
                }
            }
            return build_model(model_cfg)
        except Exception:
            pass

        # Fallback: torchvision
        import torchvision
        if hasattr(torchvision.models, model_type):
            pretrained = getattr(self.config.model, "pretrained_path", None)
            return getattr(torchvision.models, model_type)(pretrained=pretrained is not None)

        raise ValueError(f"unknown model: {model_type!r}. Provide build_model_fn.")