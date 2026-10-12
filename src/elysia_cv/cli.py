"""ElysiaCV CLI entry point — adapted from ElysiaRobot cli.py.

Provides the ``elysia`` command with sub-commands: train, test, infer, models.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
import torch.nn as nn


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
                              help="override config values (e.g. --set data.batch_size=64 optimizer.lr=0.01)")

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


def _cmd_train(args: argparse.Namespace) -> None:
    """Train sub-command — full training pipeline."""
    from elysia_cv.config import load_cv_config_with_object
    from elysia_cv.training.plan import TrainingPlan

    # 1. Load and validate config
    cfg = load_cv_config_with_object(args.config)
    if args.set:
        cfg = _apply_cli_overrides(cfg, args.set)

    # 2. Compile training plan
    plan = TrainingPlan.from_config(cfg)

    # 3. Setup GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = ""
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"

    # 4. Build model
    model = _build_model_from_config(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 5. Build dataloader
    train_loader, val_loader = _build_dataloaders(cfg)

    # 6. Build optimizer + scheduler
    from elysia_cv.optim import build_optimizer, build_scheduler
    opt_cfg = {"type": cfg.optim.type, "params": dict(cfg.optim.params)}
    optimizer = build_optimizer(model, opt_cfg)
    sched_cfg = {"type": cfg.schedule.type, "steps": cfg.schedule.steps,
                 "base_lr": cfg.schedule.base_lr, "min_lr": cfg.schedule.min_lr,
                 "warmup_steps": cfg.schedule.warmup_steps}
    scheduler = build_scheduler(optimizer, sched_cfg)

    total_steps = cfg.schedule.steps
    grad_accum = getattr(cfg.engine.optimization, "gradient_accumulation_steps", 1)
    grad_clip = getattr(cfg.engine.optimization, "grad_clip_norm", None)
    log_freq = getattr(cfg.tracking, "log_freq", 50)
    val_freq = getattr(cfg.eval, "eval_freq", 1000)
    eff_batch = cfg.data.train.batch_size * grad_accum

    # Determine image size
    bb_cfg = dict(cfg.model.backbone)
    img_size = bb_cfg.get("input_size", [448, 448])
    if isinstance(img_size, (list, tuple)):
        img_size = img_size[0]

    # ══════════════════════════════════════════════════════════
    #  Training Header
    # ══════════════════════════════════════════════════════════
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
        ("║" + _row("Optimizer: ", cfg.optim.type, "LR:", f"{sched_cfg['base_lr']:.2e}").ljust(W) + "║"),
        ("║" + _row("Scheduler: ", cfg.schedule.type, "Steps:", f"{total_steps:,}").ljust(W) + "║"),
        ("║" + _row("Batch:     ", str(eff_batch), "Img size:", f"{img_size}x{img_size}").ljust(W) + "║"),
    ]
    dev_str = f"{device} ({gpu_name})" if device.type == "cuda" and gpu_name else str(device)
    lines.append(("║" + _row("Device:    ", dev_str).ljust(W) + "║"))
    lines.append(("╚" + "═" * W + "╝"))

    for line in lines:
        print(line, flush=True)
    print("", flush=True)

    # ══════════════════════════════════════════════════════════
    #  Training Loop
    # ══════════════════════════════════════════════════════════
    from elysia_cv.models.classification.cassava.runtime import forward_train
    from elysia_cv.core.registry import LOSSES

    def _run_validation(model_: nn.Module, loader_, device_) -> tuple[float, float]:
        """Run one validation pass. Returns (loss, accuracy)."""
        model_.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loss_fn = LOSSES.build({"type": "FocalLoss", "alpha": 0.25, "gamma": 2.0})

        with torch.no_grad():
            for batch_ in loader_:
                imgs_ = batch_["inputs"].to(device_, non_blocking=True)
                labels_ = batch_["gt_label"].to(device_, non_blocking=True)
                logits_ = model_(imgs_)["logits"]
                total_loss += loss_fn(logits_, labels_).item() * len(labels_)
                total_correct += (logits_.argmax(dim=1) == labels_).sum().item()
                total_samples += len(labels_)

        model_.train()
        return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)

    is_tty = sys.stdout.isatty()

    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    model.train()
    data_iter = iter(train_loader)
    global_step = 0
    best_acc = 0.0
    loss_window: list[float] = []

    # ── TTY mode: progress bar ─────────────────────────────────
    pbar = None
    if is_tty and _has_tqdm:
        bar_fmt = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        )
        pbar = tqdm(total=total_steps, desc="Training", unit="step",
                     bar_format=bar_fmt, dynamic_ncols=True,
                     postfix={"loss": "---", "lr": "---"})
    else:
        # Non-TTY (piped/tee): plain log line
        print(f"  Training {total_steps:,} steps, log every {log_freq} steps", flush=True)

    try:
        while global_step < total_steps:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                imgs = batch["inputs"].to(device, non_blocking=True)
                labels = batch["gt_label"].to(device, non_blocking=True)

                result = forward_train(model, {"inputs": imgs, "gt_label": labels})
                loss = result["loss"] / grad_accum
                loss.backward()
                accum_loss += loss.item()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            loss_window.append(accum_loss)
            lr = scheduler.get_last_lr()[0]

            # ── Progress bar update ────────────────────────────
            if pbar is not None:
                pbar.update(1)
                if global_step % log_freq == 0:
                    avg = sum(loss_window[-log_freq:]) / len(loss_window[-log_freq:])
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")
                elif global_step <= 5:
                    pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}")

            # ── Logging (non-TTY mode) ─────────────────────────
            if pbar is None and (global_step % log_freq == 0 or global_step == 1):
                avg = sum(loss_window[-log_freq:]) / min(len(loss_window), log_freq)
                pct = 100.0 * global_step / total_steps
                print(f"  [{pct:5.1f}%] step {global_step:>6,d}/{total_steps:,d}  "
                      f"loss={avg:.4f}  lr={lr:.2e}", flush=True)

            # ── Validation ──────────────────────────────────────
            if global_step % val_freq == 0:
                val_loss, val_acc = _run_validation(model, val_loader, device)
                improved = val_acc > best_acc
                if improved:
                    best_acc = val_acc
                    best_path = f"{cfg.run.output_dir}/checkpoint_best.pt"
                    os.makedirs(cfg.run.output_dir, exist_ok=True)
                    torch.save({"step": global_step, "model": model.state_dict(),
                                 "acc": best_acc}, best_path)

                flag = " *" if improved else ""
                msg = f"  [eval]  step {global_step:>6,d}  acc={val_acc:.4f}  val_loss={val_loss:.4f}{flag}"
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg, flush=True)

            # ── Checkpoint ─────────────────────────────────────
            if global_step % val_freq == 0:
                ckpt_name = f"checkpoint_step_{global_step}.pt"
                ckpt_path = f"{cfg.run.output_dir}/{ckpt_name}"
                os.makedirs(cfg.run.output_dir, exist_ok=True)
                torch.save({"step": global_step, "model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict()}, ckpt_path)
                msg = f"  [checkpoint] {ckpt_name}"
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg, flush=True)

    finally:
        if pbar is not None:
            pbar.close()

    # ══════════════════════════════════════════════════════════
    #  Final save
    # ══════════════════════════════════════════════════════════
    final_path = f"{cfg.run.output_dir}/checkpoint_final.pt"
    torch.save({"step": global_step, "model": model.state_dict()}, final_path)

    print("╔" + "═" * W + "╗", flush=True)
    print("║" + "  Training Complete".ljust(W) + "║", flush=True)
    print("║" + f"  Steps:      {global_step:,}".ljust(W) + "║", flush=True)
    print("║" + f"  Best Acc:   {best_acc:.4f}".ljust(W) + "║", flush=True)
    print("║" + f"  Checkpoint: {final_path}".ljust(W) + "║", flush=True)
    print("╚" + "═" * W + "╝", flush=True)


def _apply_cli_overrides(cfg: Any, overrides: list[str]) -> Any:
    """Apply --set KEY=VALUE overrides to a frozen config object.

    Since CVConfig is frozen, we return the modified config (mutating
    via object.__setattr__ to bypass frozen restriction).
    """
    for ov in overrides:
        if "=" not in ov:
            print(f"  WARNING: ignoring malformed override: {ov}")
            continue
        key, value = ov.split("=", 1)
        value = _coerce_value(value)

        parts = key.split(".")
        obj = cfg
        for i, part in enumerate(parts[:-1]):
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
    if raw.lower() == "null" or raw.lower() == "none":
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


def _build_model_from_config(cfg: Any, device: torch.device) -> "nn.Module":
    """Build model using the cassava runtime (supports MLP/CNN/ResNet/ViT)."""
    from elysia_cv.models.classification.cassava.runtime import build_model

    # Convert config sections to dict for build_model
    model_cfg = {
        "model": {
            "type": cfg.model.type,
            "backbone": dict(cfg.model.backbone),
            "head": dict(cfg.model.head),
        }
    }
    model = build_model(model_cfg)
    return model.to(device)


def _build_dataloaders(cfg: Any):
    """Build train + val dataloaders from config.

    Determines image size from model backbone config.
    MLP → 64, CNN/ViT/ResNet → 448.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from elysia_cv.data import CassavaDataset

    train_cfg = cfg.data.train
    val_cfg = cfg.data.val

    # Determine input size from backbone
    backbone_cfg = dict(cfg.model.backbone)
    bb_type = backbone_cfg.get("type", "ResNet")
    if bb_type == "MLP":
        input_size = backbone_cfg.get("input_size", [64, 64])
        if isinstance(input_size, (list, tuple)):
            img_size = input_size[0] if input_size[0] == input_size[1] else 64
        else:
            img_size = input_size
    else:
        img_size = 448

    train_transform = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CassavaDataset(train_cfg.dataset_root, split=train_cfg.split,
                               transform=train_transform)
    val_ds = CassavaDataset(val_cfg.dataset_root, split=val_cfg.split,
                             transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory,
        persistent_workers=train_cfg.persistent_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_cfg.batch_size, shuffle=False,
        num_workers=val_cfg.num_workers, pin_memory=val_cfg.pin_memory,
    )
    return train_loader, val_loader


def _cmd_test(args: argparse.Namespace) -> None:
    """Stub: test sub-command."""
    print(f"[elysia test] config={args.config} checkpoint={args.checkpoint}")
    print("  Evaluation not yet implemented (coming in P5+).")


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