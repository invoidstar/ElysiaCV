# AGENTS.md — ElysiaCV

> AI agents and contributors: read this first before modifying the codebase.
> Last updated: 2026-10-12

---

## 1. What is ElysiaCV?

A unified training framework for computer vision (classification, detection, segmentation).
The guiding principle: **"Focus on models, reuse engineering."**

- **Config-driven**: everything flows from YAML configs (MMDetection-style `_base_` inheritance)
- **Plugin architecture**: models register themselves via decorators, no central registry file
- **Component library**: shared backbones, necks, heads, losses, metrics — mix and match
- **Training engine**: TrainerLoop + HookBus (checkpoint, logging, evaluation), DDP-ready

**Version**: v0.1.0 Alpha | **Python**: ≥3.10 | **PyTorch**: ≥2.0

---

## 2. Project Layout

```
ElysiaCV/
├── configs/                          # ⬅️ START HERE for experiments
│   ├── _base_/default_runtime.yaml   #    engine, tracking, dataloader defaults
│   ├── models/                       #    pure model definitions (arch + optim + schedule)
│   │   ├── mlp.yaml                  #    MLP: 64×64, AdamW
│   │   ├── simple_cnn.yaml           #    5-stage plain CNN, SGD
│   │   ├── resnet{18,34,50}.yaml     #    ResNet, SGD
│   │   └── vit_{b16,l16}.yaml        #    ViT, AdamW + grad_clip
│   ├── datasets/                     #    pure dataset definitions (data section only)
│   │   └── cassava.yaml              #    Cassava leaf disease (5 classes, 17k imgs)
│   └── experiments/                  #    composition: model + dataset + overrides
│       └── cassava_{model}.yaml      #    _base_: [../_base_/runtime, ../models/X, ../datasets/cassava]
│
├── src/elysia_cv/                    # Python package
│   ├── core/                         # 0-dependency foundation
│   │   ├── registry.py              #    REGISTRY = Registry("...") + @REGISTRY.register_module()
│   │   ├── plugin.py                #    PluginBase (model contract)
│   │   ├── api_version.py           #    CORE_API_VERSION = "1.0"
│   │   ├── errors.py                #    Custom error types
│   │   ├── capabilities.py          #    Capability enum
│   │   └── readiness.py             #    Health check
│   ├── config/                       # Config loading & validation
│   │   ├── loader.py                #    load_cv_config() — YAML + _base_ + env expansion
│   │   ├── config.py                #    CVConfig dataclass
│   │   ├── sections.py              #    Dataclass sections (Run, Model, Data, Engine, etc.)
│   │   └── validation.py            #    Config schema validation
│   ├── models/                       # Task-specific model plugins
│   │   ├── contract.py              #    Abstract ModelPlugin contract
│   │   ├── registry.py              #    MODEL_REGISTRY
│   │   └── classification/cassava/  #    Cassava disease classification
│   │       ├── plugin.py            #    @register_model decorator entry
│   │       └── runtime.py           #    build_model(), forward_train(), forward_test()
│   ├── components/                   # Shared building blocks (Registry-registered)
│   │   ├── backbones/               #    ResNet, ViT, MLP, SimpleCNN
│   │   ├── necks/                   #    FPN, IdentityNeck
│   │   └── heads/                   #    LinearClsHead, DetHead, SegHead
│   ├── training/                     # Training infrastructure
│   │   ├── loop.py                  #    TrainerLoop — main training orchestrator
│   │   ├── engine.py                #    TrainingEngine — step-level logic
│   │   ├── plan.py                  #    TrainingPlan — pre-flight validation
│   │   ├── strategy.py              #    ExecutionStrategy (grad_accum, grad_clip)
│   │   ├── distributed.py           #    DDP, FSDP support
│   │   └── hooks/                   #    Hook system (base, checkpoint, logger, evaluation, lr_scheduler)
│   ├── data/                         # Data loading (currently used by CLI, not TrainerLoop)
│   │   ├── cassava_dataset.py       #    CassavaDataset (torch Dataset)
│   │   ├── dataset.py               #    Base dataset abstraction
│   │   ├── dataloader.py            #    Dataloader builder
│   │   └── transforms/              #    Augmentation, normalization, formatting
│   ├── losses/                       # Loss functions (FocalLoss, CrossEntropy, Dice, SmoothL1)
│   ├── metrics/                      # Metrics (Accuracy, TopK, IoU, MeanAP placeholder)
│   ├── optim/                        # Optimizer & scheduler builders
│   ├── cli.py                        # CLI entry: `elysia train|test|infer|models`
│   └── __init__.py                   # Auto-registers all components on import
│
├── tools/                            # Standalone scripts
│   ├── train.py                      #   `python tools/train.py <config> [--set ...]`
│   ├── analyze.py                    #   Post-training: confusion matrix + GradCAM
│   ├── test.py                       #   Placeholder
│   └── infer.py                      #   Placeholder
│
├── scripts/                          # Batch experiment launchers
│   └── run_cassava_experiments.sh    #   Run all 7 cassava models sequentially
│
├── tests/                            # pytest — 7 test files, 121+ total tests
│   ├── test_cassava.py               #   Cassava model building + forward
│   ├── test_cassava_experiments.py   #   Full 10-step train + config tests
│   ├── test_config.py                #   Config loading + _base_ inheritance + validation
│   ├── test_det_seg.py               #   Detection & segmentation model building
│   ├── test_plugin.py                #   Plugin discovery
│   ├── test_registry.py              #   Registry register + build
│   └── test_training.py              #   TrainerLoop, HookBus, distributed
│
└── docs/                             # Guides and architecture docs
    ├── guides/                       #   config.md, training.md, extension.md
    ├── infrastructure/               #   architecture.md, performance.md
    └── status/current.md            #   What works, what doesn't (keep updated)
```

---

## 3. Architecture Principles

### 3.1 Module Dependency Direction

```
core/  ←  config/  ←  data/  ←  models/  ←  training/
  ↓         ↓          ↓         ↓            ↓
  [ALL modules can import from core/ and config/]
  
components/  ←  used by models/ (via Registry, not direct import)
losses/      ←  used by models/runtime.py
metrics/     ←  used by training/
optim/       ←  used by cli.py / training/
```

**Rules**:
- `core/` imports **zero torch code**, zero project code
- `config/` only imports `core/`
- `components/` register via `@BACKBONES.register_module()`, never import from `models/`
- `models/*/runtime.py` owns **forward pass logic** (FocalLoss, accuracy)
- `training/loop.py` owns **the training loop** (step, backward, optimizer)
- `cli.py` is the **only entry point** that wires model + data + training together

### 3.2 Config Composition Model

```
models/xxx.yaml          datasets/xxx.yaml
       \                      /
        \     _base_          /
         └───→ experiment.yaml ←─── _base_/default_runtime.yaml
```

- `_base_` supports lists: `_base_: [../a.yaml, ../b.yaml]` — merged left-to-right, child overrides
- `_deep_merge` recursively merges nested dicts; lists use index-based merge
- Environment variables: `${VAR}` and `${VAR:-default}` in string values

### 3.3 Registry Pattern

All components are discovered by decorator, never by import:

```python
from elysia_cv.core.registry import BACKBONES, HEADS, LOSSES

@BACKBONES.register_module()
class ResNet(nn.Module):
    ...

# Usage
backbone = BACKBONES.build({"type": "ResNet", "depth": 50})
```

Registries: `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`, `METRICS`, `TRANSFORMS`, `HOOKS`

---

## 4. Key Files to Know

| File | Purpose | When to edit |
|------|---------|-------------|
| `src/elysia_cv/cli.py` | Main training loop (currently direct, not via TrainerLoop) | Add new training features |
| `src/elysia_cv/config/loader.py` | `_load_yaml_with_base()` — config inheritance | Fix config loading bugs |
| `src/elysia_cv/models/classification/cassava/runtime.py` | `build_model()` + `forward_train()` | Change Cassava model forward |
| `src/elysia_cv/training/loop.py` | `TrainerLoop.run()` — full training orchestrator | Fix training loop bugs |
| `src/elysia_cv/training/hooks/` | HookBus + built-in hooks | Add checkpoint/log/eval hooks |
| `src/elysia_cv/__init__.py` | Auto-registers all components on import | Add new component modules |
| `configs/_base_/default_runtime.yaml` | Global defaults (engine, tracking, dataloader) | Change system-wide defaults |
| `pyproject.toml` | Package metadata, dependencies, lint config | Add new dependencies |

---

## 5. Development Workflow

### 5.1 Setup

```bash
conda activate CV
pip install -e ".[dev]"
```

### 5.2 Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_cassava_experiments.py -v

# With GPU (CI-friendly)
CUDA_VISIBLE_DEVICES=6 pytest tests/ -v
```

### 5.3 Lint

```bash
ruff check src/ tests/ tools/
black --check src/ tests/ tools/
```

Config: 120-char lines, py310 target, rules `A,B,E,F,I,W`.

### 5.4 Run an Experiment

```bash
# Single model
python tools/train.py configs/experiments/cassava_resnet50.yaml

# With overrides
python tools/train.py configs/experiments/cassava_vit_b16.yaml --set schedule.steps=100 optim.params.lr=0.001

# Full suite
bash scripts/run_cassava_experiments.sh
```

### 5.5 Post-Training Analysis

```bash
python tools/analyze.py configs/experiments/cassava_resnet50.yaml \
    --checkpoint work_dirs/cassava_resnet50/checkpoint_best.pt
```

Outputs: `analysis/confusion_matrix.png`, `analysis/per_class_report.txt`, `analysis/gradcam/`

---

## 6. Adding a New Model

1. **Create model config**: `configs/models/my_model.yaml`
   - Define `model`, `optim`, `schedule`, `engine.optimization`

2. **Create runtime**: `src/elysia_cv/models/classification/my_model/`
   - `plugin.py` — `@register_model` decorator
   - `runtime.py` — `build_model()`, `forward_train()`, `forward_test()`

3. **Create experiment**: `configs/experiments/my_experiment.yaml`
   - `_base_: [../_base_/default_runtime.yaml, ../models/my_model.yaml, ../datasets/my_data.yaml]`

4. **Add components** (if new backbone/head/neck):
   - Register with `@BACKBONES.register_module()` / `@HEADS.register_module()`
   - Import module in `src/elysia_cv/__init__.py:_register_default_components()`

5. **Add tests**: `tests/test_my_model.py`
   - Build test, forward test, 10-step train test

---

## 7. Adding a New Dataset

1. **Create dataset config**: `configs/datasets/my_data.yaml`
   - Define `data.train` and `data.val` sections
   - `_build_dataloaders()` in `cli.py` auto-generates transforms based on model backbone type

2. **Create experiment** (or reuse existing):
   - `_base_: [../_base_/default_runtime.yaml, ../models/X.yaml, ../datasets/my_data.yaml]`

---

## 8. Known Issues & Current State

| Issue | Status | Detail |
|-------|--------|--------|
| TrainerLoop not used in CLI | ⚠️ | `cli.py` uses a direct training loop, bypassing `TrainerLoop`/`HookBus`. Reason: interface mismatches were causing repeated breakage. Plan: refactor after v0.1. |
| `test.py` / `infer.py` | ⚠️ | Placeholders only |
| Detection runtime | ⚠️ | Uses random pseudo-labels |
| MeanAP metric | ⚠️ | Placeholder |
| `data/` module | ⚠️ | Files exist but not fully wired into TrainerLoop |
| `CLI` output_dir | ✅ | Relative `./work_dirs/` resolved at runtime from CWD |
| ViT-L OOM | ✅ | Fixed: batch=4 + grad_accum=8 |

**See** `docs/status/current.md` for the canonical status.

---

## 9. Code Conventions

- **Type hints**: `from __future__ import annotations` at top of every `.py`
- **Docstrings**: Google-style, triple-quoted, every public class/function
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- **Imports**: stdlib → third-party → internal, each group separated by blank line
- **No circular imports**: use `from __future__ import annotations` + `TYPE_CHECKING` pattern when needed
- **GPU**: always use `device` parameter, never hardcode `"cuda:0"`; use `non_blocking=True` for data transfers
- **Config access**: use `cfg.section.field` (dataclass attributes), never `cfg["section"]["field"]`
- **Print to stdout**: always use `flush=True` when output is piped/captured
- **Error handling**: raise from `elysia_cv.core.errors`, not bare `Exception`

---

## 10. Commit Conventions

```
feat: new feature
fix: bug fix
refactor: code restructuring (no behavior change)
docs: documentation only
test: test additions/changes
chore: build, CI, dependencies
style: formatting, linting (no code change)
```

Examples:
- `feat: add ViT-L/16 support with grad_accum=8`
- `fix: save_freq→val_freq rename in checkpoint section`
- `refactor: split model/dataset configs for composability`
- `docs: update training guide with experiments/ path`