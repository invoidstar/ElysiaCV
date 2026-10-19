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
    train_parser.add_argument("--resume", "-r", nargs="?", const="auto", default=None,
                              metavar="PATH",
                              help="resume from checkpoint (auto-discover latest if no path given)")
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

    # config (dump resolved config)
    cfg_parser = sub.add_parser("config", help="dump resolved config")
    cfg_parser.add_argument("action", nargs="?", default="dump", choices=["dump"],
                            help="config sub-action (default: dump)")
    cfg_parser.add_argument("config", help="path to YAML config file")
    cfg_parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                            help="override config values before dumping")

    # models (list registered models)
    sub.add_parser("models", help="list available model plugins")

    args = parser.parse_args(argv)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "test":
        _cmd_test(args)
    elif args.command == "infer":
        _cmd_infer(args)
    elif args.command == "config":
        _cmd_config_dump(args)
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

    # 1.5. Resolve resume checkpoint path
    resume_from = None
    if args.resume is not None:
        resume_from = _resolve_resume_path(args.resume, cfg.run.output_dir)

    # 2. Print training header (rank 0 only in DDP)
    _is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if _is_main:
        _print_training_header(cfg, resume_from=resume_from)

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
    engine.run(resume_from=resume_from)


def _resolve_resume_path(path_or_auto: str | None, output_dir: str) -> str | None:
    """Resolve ``--resume`` to an actual checkpoint path.

    - ``"auto"`` (or empty): discover the latest checkpoint in output_dir.
    - explicit path: use as-is.
    """
    import glob
    output = str(output_dir)

    if path_or_auto == "auto" or not path_or_auto:
        # Try in order: checkpoint_final → checkpoint_best_* → checkpoint_step_*
        patterns = [
            os.path.join(output, "checkpoint_final.pt"),
            os.path.join(output, "checkpoint_best_*.pt"),
            os.path.join(output, "checkpoint_step_*.pt"),
        ]
        for pat in patterns:
            matches = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
            if matches:
                return matches[0]
        print(f"[resume] WARNING: no checkpoint found in {output}")
        return None

    if os.path.exists(path_or_auto):
        return path_or_auto
    print(f"[resume] WARNING: checkpoint not found: {path_or_auto}")
    return None


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
    """Return a callable that builds the model via plugin discovery."""
    def _build():
        return _build_model_from_config(cfg)
    return _build


def _make_build_loader_fn(cfg: Any, split: str):
    """Return a callable that builds a DataLoader from config.

    Uses the ElysiaCV TRANSFORMS registry pipeline when
    ``data.{split}.transforms`` is configured, falling back to a
    sensible torchvision.Compose default for backward compatibility.
    """
    from torch.utils.data import DataLoader
    from elysia_cv.data import CassavaDataset
    from elysia_cv.data.dataloader import build_dataloader

    def _build():
        data_cfg = cfg.data.train if split == "train" else cfg.data.val

        # Determine image size from backbone type
        bb_cfg = dict(cfg.model.backbone)
        bb_type = bb_cfg.get("type", "ResNet")
        if bb_type == "MLP":
            img_size_raw = bb_cfg.get("input_size", [64, 64])
        else:
            img_size_raw = bb_cfg.get("input_size", [448, 448])
        if isinstance(img_size_raw, (list, tuple)):
            img_size_raw = img_size_raw[0] if img_size_raw[0] == img_size_raw[1] else 448

        # Build transform pipeline — prefer registry, fall back to torchvision
        transforms_cfg = getattr(data_cfg, "transforms", None)
        if transforms_cfg is not None and len(transforms_cfg) > 0:
            from elysia_cv.data.transforms import build_pipeline
            pipeline = build_pipeline(list(transforms_cfg))
        else:
            # Legacy fallback
            from torchvision import transforms as T
            import torch
            if split == "train":
                pipeline = T.Compose([
                    T.Resize(int(img_size_raw * 1.14)),
                    T.RandomResizedCrop(img_size_raw, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(0.5),
                    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                pipeline = T.Compose([
                    T.Resize(img_size_raw),
                    T.CenterCrop(img_size_raw),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        ds = CassavaDataset(
            data_cfg.dataset_root,
            split=data_cfg.split,
            pipeline=pipeline,
        )
        return build_dataloader(
            ds,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            prefetch_factor=getattr(data_cfg, "prefetch_factor", 2),
            pin_memory=data_cfg.pin_memory,
            persistent_workers=data_cfg.persistent_workers if split == "train" else False,
            drop_last=getattr(data_cfg, "drop_last", True) if split == "train" else False,
            shuffle=(split == "train"),
        )

    return _build


# ═══════════════════════════════════════════════════════════════════
#  Training header (box-style, matches old inline-loop output)
# ═══════════════════════════════════════════════════════════════════

def _print_training_header(cfg: Any, resume_from: str | None = None) -> None:
    """Print a box-style training header with run metadata."""
    # Quick param count (CPU, no grad)
    try:
        model = _build_model_from_config(cfg)
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

    if resume_from:
        print(f"  [resume] Restoring from: {resume_from}\n", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  test / infer commands
# ═══════════════════════════════════════════════════════════════════

def _cmd_test(args: argparse.Namespace) -> None:
    """Evaluate a trained model on the validation set."""
    import torch
    from elysia_cv.config import load_cv_config_with_object

    cfg = load_cv_config_with_object(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = _build_model_from_config(cfg)
    model = model.to(device)
    model.eval()

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        # Auto-discover best checkpoint
        output_dir = cfg.run.output_dir
        candidates = [
            f"{output_dir}/checkpoint_best_Accuracy.pt",
            f"{output_dir}/checkpoint_best.pt",
            f"{output_dir}/checkpoint_final.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found. Use --checkpoint PATH.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")

    # Build val dataloader
    val_loader = _make_build_loader_fn(cfg, split="val")()

    # Build metrics for evaluation
    from elysia_cv.core.registry import METRICS
    metric_names = getattr(cfg.eval, "metrics", [{"type": "Accuracy"}])
    if isinstance(metric_names, (list, tuple)):
        if metric_names and isinstance(metric_names[0], str):
            # Legacy: ["Accuracy", "Top5Accuracy"] format
            metrics = [METRICS.build({"type": m}) for m in metric_names]
        else:
            metrics = [METRICS.build(m) for m in metric_names]
    else:
        metrics = [METRICS.build({"type": "Accuracy"})]

    # Run evaluation
    with torch.no_grad():
        for batch in val_loader:
            _eval_step(batch, model, device, metrics, cfg)

    # Report results
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    _report_metrics(metrics)
    print("=" * 50)


def _eval_step(
    batch: dict,
    model: torch.nn.Module,
    device: torch.device,
    metrics: list,
    cfg: Any,
) -> None:
    """One evaluation step — routes to classification or detection path."""
    model_type = str(getattr(cfg.model, "type", ""))

    if "fcos" in model_type or "detection" in model_type:
        # Detection: use forward_test + decode + per-image update
        from elysia_cv.models.detection.fcos.runtime import forward_test

        outputs = forward_test(model, batch, device=device)
        preds = outputs.get("pred_instances", [])
        gt_bboxes = batch.get("gt_bboxes", [])
        gt_labels = batch.get("gt_labels", [])

        for i in range(len(preds)):
            for m in metrics:
                try:
                    m.update(
                        pred_boxes=preds[i]["bboxes"],
                        pred_scores=preds[i]["scores"],
                        pred_labels=preds[i]["labels"],
                        gt_boxes=gt_bboxes[i],
                        gt_labels=gt_labels[i],
                    )
                except Exception:
                    pass
        return

    # Classification (default)
    if isinstance(batch, dict) and "inputs" in batch:
        inputs = batch["inputs"].to(device, non_blocking=True)
        labels = batch.get("gt_label", batch.get("labels"))
        if labels is not None:
            labels = labels.to(device, non_blocking=True)
    else:
        inputs = batch.to(device)
        labels = None

    outputs = model(inputs)
    if isinstance(outputs, dict):
        logits = outputs.get("logits", outputs)
    else:
        logits = outputs

    for m in metrics:
        if labels is not None:
            m.update(logits, labels)
        else:
            m.update(logits)


def _report_metrics(metrics: list) -> None:
    """Print metric results."""
    for m in metrics:
        try:
            val = m.compute()
            name = m.__class__.__name__
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"  {k:<20s}: {float(v):.4f}")
            else:
                print(f"  {name:<20s}: {float(val):.4f}")
        except Exception as exc:
            print(f"  {m.__class__.__name__:<20s}: (error: {exc})")


def _cmd_infer(args: argparse.Namespace) -> None:
    """Run inference on a single image or directory of images."""
    import json
    from pathlib import Path
    import torch
    from PIL import Image
    from elysia_cv.config import load_cv_config_with_object

    cfg = load_cv_config_with_object(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = _build_model_from_config(cfg)
    model = model.to(device)
    model.eval()

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        output_dir = cfg.run.output_dir
        for name in ["checkpoint_best_Accuracy.pt", "checkpoint_best.pt", "checkpoint_final.pt"]:
            cand = f"{output_dir}/{name}"
            if os.path.exists(cand):
                ckpt_path = cand
                break
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("ERROR: checkpoint not found. Use --checkpoint PATH.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")

    # Determine transform
    from torchvision import transforms as T
    import torch
    bb_cfg = dict(cfg.model.backbone)
    img_size_raw = bb_cfg.get("input_size", [448, 448])
    if isinstance(img_size_raw, (list, tuple)):
        img_size_raw = img_size_raw[0]
    transform = T.Compose([
        T.Resize(img_size_raw),
        T.CenterCrop(img_size_raw),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect input images
    input_path = Path(args.input)
    if input_path.is_dir():
        image_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    else:
        image_files = [input_path]

    if not image_files:
        print("No image files found.")
        sys.exit(1)

    results = []
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        probs = torch.softmax(logits, dim=1)[0]
        top5_values, top5_indices = probs.topk(min(5, probs.size(0)))

        entry = {
            "image": str(img_file),
            "predictions": [
                {"class": int(idx), "confidence": float(val)}
                for idx, val in zip(top5_indices, top5_values)
            ],
        }
        results.append(entry)

        # Print
        print(f"\n{img_file.name}")
        for rank, (idx, val) in enumerate(zip(top5_indices, top5_values)):
            marker = " *" if rank == 0 else "  "
            print(f"  {marker} class_{int(idx)}: {float(val):.4f}")

    # Save JSON if processing many files
    if len(results) > 1:
        out_json = "inference_results.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {out_json}")


# ═══════════════════════════════════════════════════════════════════
#  Generic model builder (decoupled from cassava)
# ═══════════════════════════════════════════════════════════════════

def _build_model_from_config(cfg: Any) -> torch.nn.Module:
    """Build a model from config using plugin discovery or torchvision fallback.

    This replaces the old hardcoded cassava-only builder.
    """
    model_type = cfg.model.type

    # 1. Detection models (fcos)
    if model_type in ("fcos",):
        from elysia_cv.models.detection.fcos.runtime import build_model
        model_cfg: dict = {
            "model": {
                "type": model_type,
                "backbone": dict(cfg.model.backbone),
            }
        }
        neck = getattr(cfg.model, "neck", None)
        if neck is not None:
            model_cfg["model"]["neck"] = dict(neck)
        head = getattr(cfg.model, "head", None)
        if head is not None:
            model_cfg["model"]["head"] = dict(head)
        pretrained = getattr(cfg.model, "pretrained_path", None)
        if pretrained:
            model_cfg["model"]["pretrained_path"] = pretrained
        return build_model(model_cfg)

    # 2. Try plugin system (ModelRegistry)
    try:
        from elysia_cv.models.registry import get_model_registry
        registry = get_model_registry()
        plugin_entry = registry.get(model_type)
        plugin = plugin_entry.plugin
        build_target = plugin.get_model_build_target()  # "module:func"
        module_path, func_name = build_target.split(":")
        import importlib
        mod = importlib.import_module(module_path)
        build_fn = getattr(mod, func_name)
        model_cfg: dict = {
            "model": {
                "type": model_type,
                "backbone": dict(cfg.model.backbone),
            }
        }
        neck = getattr(cfg.model, "neck", None)
        if neck is not None:
            model_cfg["model"]["neck"] = dict(neck)
        head = getattr(cfg.model, "head", None)
        if head is not None:
            model_cfg["model"]["head"] = dict(head)
        pretrained = getattr(cfg.model, "pretrained_path", None)
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
                "backbone": dict(cfg.model.backbone),
                "head": dict(getattr(cfg.model, "head", {})),
            }
        }
        return build_model(model_cfg)
    except Exception:
        pass

    # 4. torchvision fallback
    import torchvision
    if hasattr(torchvision.models, model_type):
        return getattr(torchvision.models, model_type)(pretrained=False)

    raise ValueError(f"Unknown model: {model_type!r}")


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


def _cmd_config_dump(args: argparse.Namespace) -> None:
    """Dump the fully-resolved config (with _base_ inheritance expanded)."""
    from elysia_cv.config import load_cv_config_with_object

    cfg = load_cv_config_with_object(args.config)
    if args.set:
        cfg = _apply_cli_overrides(cfg, args.set)

    d = cfg.dump()

    # Try YAML first for readability
    try:
        import yaml
        print(yaml.dump(d, default_flow_style=False, sort_keys=False, allow_unicode=True))
    except ImportError:
        import json
        print(json.dumps(d, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()