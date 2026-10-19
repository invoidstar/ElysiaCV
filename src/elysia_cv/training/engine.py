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

    def run(self, resume_from: str | None = None) -> None:
        """Full training pipeline: init → build → loop → cleanup.

        Args:
            resume_from: Path to a checkpoint to resume from.
        """
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

        # Wrap with distributed strategy (DDP or FSDP)
        strategy = self.config.engine.strategy
        if strategy == "fsdp":
            from elysia_cv.training.distributed import wrap_fsdp
            model = wrap_fsdp(model, self._dist_ctx)
        else:
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

        # 5. Build AMP / precision context
        precision = self.config.engine.precision
        autocast_ctx, grad_scaler = _build_autocast(precision, device)

        # 6. Build strategy
        strategy = ExecutionStrategy(
            grad_accum_steps=self.config.engine.optimization.gradient_accumulation_steps,
            grad_clip_norm=self.config.engine.optimization.grad_clip_norm,
        )

        # 7. Build hooks (order matters: Checkpoint → Logger for final summary)
        ckpt_hook = CheckpointHook(
            output_dir=self.config.run.output_dir,
            save_freq=self.config.eval.eval_freq,
            save_best_metric=self.config.eval.save_best,
            save_final=self.config.eval.save_final,
        )
        hooks = HookBus()
        hooks.register(ckpt_hook)
        hooks.register(LrSchedulerHook())
        hooks.register(LoggerHook(
            log_freq=self.config.tracking.log_freq,
            total_steps=self.config.schedule.steps,
        ))
        if val_loader is not None:
            hooks.register(EvalHook(
                dataloader=val_loader,
                eval_freq=self.config.eval.eval_freq,
                metrics=self.config.eval.metrics,
                save_best_metric=self.config.eval.save_best,
            ))

        # 8. Build cadence
        cadence = CadenceController(
            log_freq=self.config.tracking.log_freq,
            eval_freq=self.config.eval.eval_freq,
        )

        # 9. Loop
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
            autocast_ctx=autocast_ctx,
            grad_scaler=grad_scaler,
        )

        # 9.5. Resume from checkpoint if requested
        if resume_from:
            restored_step = _resume_from_checkpoint(resume_from, model, optimizer, scheduler, loop)
            if self._dist_ctx.is_main_process:
                print(f"  [resume] Restored step {restored_step}, starting from {loop.state.global_step}",
                      flush=True)

        # 10. Run
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
        return _build_loader_from_config(self.config.data.train, self.config)

    def _build_val_loader(self) -> Any | None:
        """Build validation DataLoader from config, or use injected function."""
        if self._build_val_loader_fn is not None:
            return self._build_val_loader_fn()
        val_cfg = self.config.data.val
        if not val_cfg.dataset_root:
            return None
        return _build_loader_from_config(val_cfg, self.config)

    def _build_model(self) -> torch.nn.Module:
        """Build model from config, or use injected function.

        Priority: injected function > plugin system > detection > cassava > torchvision.
        """
        if self._build_model_fn is not None:
            return self._build_model_fn()

        model_type = self.config.model.type

        # 1. Detection models (fcos)
        if model_type in ("fcos",):
            from elysia_cv.models.detection.fcos.runtime import build_model
            model_cfg: dict = {
                "model": {
                    "type": model_type,
                    "backbone": dict(self.config.model.backbone),
                }
            }
            neck = getattr(self.config.model, "neck", None)
            if neck is not None:
                model_cfg["model"]["neck"] = dict(neck)
            head = getattr(self.config.model, "head", None)
            if head is not None:
                model_cfg["model"]["head"] = dict(head)
            pretrained = getattr(self.config.model, "pretrained_path", None)
            if pretrained:
                model_cfg["model"]["pretrained_path"] = pretrained
            return build_model(model_cfg)

        # 2. Try plugin system (ModelRegistry)
        try:
            from elysia_cv.models.registry import get_model_registry
            registry = get_model_registry()
            plugin_entry = registry.get(model_type)
            plugin = plugin_entry.plugin
            build_target = plugin.get_model_build_target()
            module_path, func_name = build_target.split(":")
            import importlib
            mod = importlib.import_module(module_path)
            build_fn = getattr(mod, func_name)
            model_cfg = {
                "model": {
                    "type": model_type,
                    "backbone": dict(self.config.model.backbone),
                }
            }
            neck = getattr(self.config.model, "neck", None)
            if neck is not None:
                model_cfg["model"]["neck"] = dict(neck)
            head = getattr(self.config.model, "head", None)
            if head is not None:
                model_cfg["model"]["head"] = dict(head)
            pretrained = getattr(self.config.model, "pretrained_path", None)
            if pretrained:
                model_cfg["model"]["pretrained_path"] = pretrained
            return build_fn(model_cfg)
        except (KeyError, Exception):
            pass

        # 3. Try legacy cassava runtime
        try:
            from elysia_cv.models.classification.cassava.runtime import build_model
            model_cfg = {
                "model": {
                    "type": model_type,
                    "backbone": dict(self.config.model.backbone),
                    "head": dict(getattr(self.config.model, "head", {})),
                }
            }
            return build_model(model_cfg)
        except Exception:
            pass

        # 4. torchvision fallback
        import torchvision
        if hasattr(torchvision.models, model_type):
            return getattr(torchvision.models, model_type)(pretrained=False)

        raise ValueError(f"unknown model: {model_type!r}. Provide build_model_fn.")


def _build_loader_from_config(data_cfg: Any, full_config: Any = None) -> Any:
    """Build a DataLoader from a DataSubSection, routing to the right dataset."""
    from elysia_cv.data.dataloader import build_dataloader

    dataset = _build_dataset(data_cfg)

    # Detection tasks need custom collate for variable-size images
    collate_fn = None
    dataset_name = getattr(data_cfg, "dataset_name", "")
    if dataset_name.upper() == "COCO":
        from elysia_cv.models.detection.fcos.runtime import (
            _collate_detection_batch,
        )
        collate_fn = _collate_detection_batch

    return build_dataloader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        prefetch_factor=getattr(data_cfg, "prefetch_factor", 2),
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
        drop_last=data_cfg.drop_last,
        shuffle=(data_cfg.split == "train" or "train" in data_cfg.split),
        collate_fn=collate_fn,
    )


def _build_dataset(data_cfg: Any) -> Any:
    """Route to the correct dataset class based on ``dataset_name``."""
    dataset_name = getattr(data_cfg, "dataset_name", "")
    transforms = list(getattr(data_cfg, "transforms", []))

    if dataset_name.upper() == "COCO":
        from elysia_cv.data import CocoDetectionDataset
        return CocoDetectionDataset(
            root=str(data_cfg.dataset_root or ""),
            split=str(data_cfg.split),
            transforms=transforms,
        )

    # Default: ImageFolderDataset
    from elysia_cv.data.dataset import ImageFolderDataset
    return ImageFolderDataset(
        root=str(data_cfg.dataset_root or ""),
        split=str(data_cfg.split),
        transforms=transforms,
    )


# ═══════════════════════════════════════════════════════════════
# Precision helpers
# ═══════════════════════════════════════════════════════════════

def _resume_from_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loop: Any,
) -> int:
    """Restore model/optimizer/scheduler/step from a checkpoint.

    Returns the restored step number.
    """
    from elysia_cv.training.distributed import unwrap_model

    ckpt = torch.load(path, map_location="cpu")
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(ckpt["model_state_dict"])

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    step = int(ckpt.get("step", 0))
    epoch = int(ckpt.get("epoch", 0))
    loop.state.global_step = step
    loop.state.epoch = epoch

    return step


def _build_autocast(precision: Any, device: torch.device) -> tuple[Any, Any]:
    """Build autocast context and optional GradScaler from config.

    Returns (autocast_ctx_or_null, grad_scaler_or_None).
    """
    if not precision.mixed_precision or device.type != "cuda" or not torch.cuda.is_available():
        from contextlib import nullcontext
        return nullcontext(), None

    try:
        if precision.dtype == "float16":
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            scaler = torch.cuda.amp.GradScaler()
            return ctx, scaler
        elif precision.dtype == "bfloat16":
            if torch.cuda.is_bf16_supported():
                ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
                return ctx, None
            else:
                # bf16 not supported on this GPU → fall back
                from contextlib import nullcontext
                return nullcontext(), None
        else:
            from contextlib import nullcontext
            return nullcontext(), None
    except Exception:
        from contextlib import nullcontext
        return nullcontext(), None