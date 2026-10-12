"""ElysiaCV CLI entry point — adapted from ElysiaRobot cli.py.

Provides the ``elysia`` command with sub-commands: train, test, infer, models.

The ``train`` command delegates to :class:`TrainingEngine`, which wires
distributed init, dataloaders, model, hooks, and the :class:`TrainerLoop`.

Task-specific logic (model building, dataset loading) is injected via
constructor callbacks so that the same engine works for any task.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch


def main(argv: list[str] | None = None) -> None:
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="elysia",
        description="ElysiaCV — unified training framework for computer vision research.",
    )
    sub = parser.add_subparsers(dest="command", help="sub-command")

    # train
    train_parser = sub.add_parser("train", help="run training")
    train_parser.add_argument("config", help="path to YAML config file")
    train_parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                              help="override config values (e.g. --set schedule.steps=100)")

    # test
    test_parser = sub.add_parser("test", help="run evaluation")
    test_parser.add_argument("config", help="path to YAML config file")
    test_parser.add_argument("--checkpoint", "-c", help="path to checkpoint")

    # infer
    infer_parser = sub.add_parser("infer", help="run inference on single image")
    infer_parser.add_argument("config", help="path to YAML config file")
    infer_parser.add_argument("--checkpoint", "-c", help="path to checkpoint")
    infer_parser.add_argument("--input", "-i", required=True, help="path to input image or directory")

    # models (list registered models)
    sub.add_parser("models", help="list available model plugins")

    args = parser.parse_args(argv)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "test":
        _cmd_test(args)
    elif args.command == "infer":
        _cmd_infer(args)
    elif args.command == "models":
        _cmd_models(args)
    else:
        parser.print_help()
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  Train command — delegates to TrainingEngine
# ═══════════════════════════════════════════════════════════════════

def _cmd_train(args: argparse.Namespace) -> None:
    """Train sub-command — load config, print header, run engine."""
    from elysia_cv.config import load_cv_config_with_object
    from elysia_cv.training.engine import TrainingEngine

    # 1. Load and validate config
    cfg = load_cv_config_with_object(args.config)
    if args.set:
        cfg = _apply_cli_overrides(cfg, args.set)

    # 2. Print training header (rank 0 only in DDP)
    _is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if _is_main:
        _print_training_header(cfg)

    # 3. Build task-specific model + dataloader functions
    build_model_fn = _make_build_model_fn(cfg)
    build_train_loader_fn = _make_build_loader_fn(cfg, split="train")
    build_val_loader_fn = _make_build_loader_fn(cfg, split="val")

    # 4. Run via TrainingEngine
    engine = TrainingEngine(
        cfg,
        build_model_fn=build_model_fn,
        build_train_loader_fn=build_train_loader_fn,
        build_val_loader_fn=build_val_loader_fn,
    )
    engine.run()


# ═══════════════════════════════════════════════════════════════════
#  CLI helpers
# ═══════════════════════════════════════════════════════════════════

def _apply_cli_overrides(cfg: Any, overrides: list[str]) -> Any:
    """Apply --set KEY=VALUE overrides to a frozen config object."""
    for ov in overrides:
        if "=" not in ov:
            print(f"  WARNING: ignoring malformed override: {ov}")
            continue
        key, value = ov.split("=", 1)
        value = _coerce_value(value)

        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        attr = parts[-1]

        try:
            object.__setattr__(obj, attr, value)
        except Exception as exc:
            print(f"  WARNING: cannot override {key}={value}: {exc}")
    return cfg


def _coerce_value(raw: str) -> Any:
    """Coerce string to int/float/bool if possible."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ═══════════════════════════════════════════════════════════════════
#  Model & dataloader builders (task-specific, injected into engine)
# ═══════════════════════════════════════════════════════════════════

def _make_build_model_fn(cfg: Any):
    """Return a callable that builds the classification model."""
    from elysia_cv.models.classification.cassava.runtime import build_model

    def _build():
        model_cfg: dict = {
            "model": {
                "type": cfg.model.type,
                "backbone": dict(cfg.model.backbone),
            }
        }
        # Optional neck (FGVC models like HERBS)
        neck = getattr(cfg.model, "neck", None)
        if neck is not None:
            model_cfg["model"]["neck"] = dict(neck)
        # Optional head (standard models)
        head = getattr(cfg.model, "head", None)
        if head is not None:
            model_cfg["model"]["head"] = dict(head)
        # Optional pretrained path
        pretrained = getattr(cfg.model, "pretrained_path", None)
        if pretrained:
            model_cfg["model"]["pretrained_path"] = pretrained
        return build_model(model_cfg)

    return _build


def _make_build_loader_fn(cfg: Any, split: str):
    """Return a callable that builds a Cassava DataLoader."""
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from elysia_cv.data import CassavaDataset

    def _build():
        data_cfg = cfg.data.train if split == "train" else cfg.data.val

        # Determine image size from backbone type
        bb_cfg = dict(cfg.model.backbone)
        bb_type = bb_cfg.get("type", "ResNet")
        if bb_type == "MLP":
            img_size = bb_cfg.get("input_size", [64, 64])
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0] if img_size[0] == img_size[1] else 64
        else:
            img_size = bb_cfg.get("input_size", [448, 448])
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0] if img_size[0] == img_size[1] else 448

        if split == "train":
            transform = T.Compose([
                T.Resize(int(img_size * 1.14)),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = T.Compose([
                T.Resize(img_size),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        ds = CassavaDataset(data_cfg.dataset_root, split=data_cfg.split, transform=transform)
        return DataLoader(
            ds,
            batch_size=data_cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=data_cfg.persistent_workers if split == "train" else False,
        )

    return _build


# ═══════════════════════════════════════════════════════════════════
#  Training header (box-style, matches old inline-loop output)
# ═══════════════════════════════════════════════════════════════════

def _print_training_header(cfg: Any) -> None:
    """Print a box-style training header with run metadata."""
    from elysia_cv.models.classification.cassava.runtime import build_model

    # Quick param count (CPU, no grad)
    try:
        model_dict: dict = {
            "model": {
                "type": cfg.model.type,
                "backbone": dict(cfg.model.backbone),
            }
        }
        neck = getattr(cfg.model, "neck", None)
        if neck is not None:
            model_dict["model"]["neck"] = dict(neck)
        head = getattr(cfg.model, "head", None)
        if head is not None:
            model_dict["model"]["head"] = dict(head)
        model = build_model(model_dict)
        total_params = sum(p.numel() for p in model.parameters())
        del model
    except Exception:
        total_params = 0

    total_steps = cfg.schedule.steps
    grad_accum = getattr(cfg.engine.optimization, "gradient_accumulation_steps", 1)
    eff_batch = cfg.data.train.batch_size * grad_accum

    bb_cfg = dict(cfg.model.backbone)
    img_size = bb_cfg.get("input_size", [448, 448])
    if isinstance(img_size, (list, tuple)):
        img_size = img_size[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = ""
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"

    W = 62

    def _row(prefix: str, val1: str, label2: str = "", val2: str = "") -> str:
        left = f"  {prefix} {val1}"
        if label2:
            right = f"{label2} {val2}"
            pad = W - len(left) - len(right)
            return left + " " * max(pad, 2) + right
        return left

    lines = [
        ("╔" + "═" * W + "╗"),
        ("║" + "  ElysiaCV Training".ljust(W) + "║"),
        ("║" + "".ljust(W) + "║"),
        ("║" + _row("Run:       ", cfg.run.name).ljust(W) + "║"),
        ("║" + _row("Model:     ", cfg.model.type, "Params:", f"{total_params:,}").ljust(W) + "║"),
        ("║" + _row("Optimizer: ", cfg.optim.type, "LR:", f"{cfg.schedule.base_lr:.2e}").ljust(W) + "║"),
        ("║" + _row("Scheduler: ", cfg.schedule.type, "Steps:", f"{total_steps:,}").ljust(W) + "║"),
        ("║" + _row("Batch:     ", str(eff_batch), "Img size:", f"{img_size}×{img_size}").ljust(W) + "║"),
    ]
    dev_str = f"{device} ({gpu_name})" if gpu_name else str(device)
    lines.append(("║" + _row("Device:    ", dev_str).ljust(W) + "║"))
    lines.append(("╚" + "═" * W + "╝"))

    for line in lines:
        print(line, flush=True)
    print("", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  Stub commands (test / infer)
# ═══════════════════════════════════════════════════════════════════

def _cmd_test(args: argparse.Namespace) -> None:
    """Stub: test sub-command."""
    print(f"[elysia test] config={args.config} checkpoint={args.checkpoint}")
    print("  Evaluation not yet implemented.")


def _cmd_infer(args: argparse.Namespace) -> None:
    """Stub: infer sub-command."""
    print(f"[elysia infer] config={args.config} input={args.input}")
    print("  Inference not yet implemented.")


def _cmd_models(args: argparse.Namespace) -> None:
    """List all discovered model plugins."""
    print("Available model plugins:")
    print("=" * 60)
    try:
        from elysia_cv.models.registry import ModelRegistry
        registry = ModelRegistry.discover()
        for name in sorted(registry.entries.keys()):
            plugin = registry.entries[name].plugin
            task = getattr(plugin, "task_type", "?")
            print(f"  {name:<30s}  [{task}]")
        print(f"\nTotal: {len(registry)} model(s)")
    except Exception as exc:
        print(f"  (discovery failed: {exc})")


if __name__ == "__main__":
    main()