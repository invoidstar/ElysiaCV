# Changelog

All notable changes to ElysiaCV will be documented in this file.

## [0.5.0] — 2026-10-21 · Phase 8: Action Recognition + Phase 9: Code Quality

### Added — Action Recognition (16 video models)

- **2D CNN + Temporal**: TSN (per-frame ResNet + temporal pool), TSM (temporal shift module)
- **3D CNN**: R3D-18, MC3-18 (mixed 2D+3D), R2Plus1D-18 (R(2+1)D), S3D (separable 3D), X3D (multi-axis expansion, XS-M)
- **Multi-scale ViT**: MViTv2 (tiny/base variants), Hiera (hierarchical attention)
- **Dual-pathway**: SlowFast (fast+slow pathway fusion)
- **CLIP-based**: X-CLIP (frozen CLIP ViT + temporal transformer), ST-Adapter (frozen ViT + 3D conv adapters)
- **CNN+ViT unified**: UniFormerV2 (early CNN + late ViT stages)
- **Transformers**: TimeSformer (divided space-time attention, K400-pretrained), VideoMAE (masked autoencoder, K400-pretrained)
- **VideoDataset**: UCF101 / Kinetics format support, Decord-based frame extraction + SampleFrames + format adapters

### Added — Infrastructure Improvements

- **model_builder.py**: Single-source-of-truth model registry (`_MODEL_REGISTRY`): 18 types in one dict, `build_model_from_config()` dispatches by family flag (video/yolo/detection). Eliminates ~420 lines of duplicated routing between `cli.py` and `engine.py`.
- **common_video_forward.py**: Shared `forward_train`/`forward_test` for all video models — eliminates copy-paste across 11 runtime.py files.
- **checkpoint_utils.py**: `find_checkpoint(output_dir)` auto-discovery (best_Accuracy → best → final priority), `load_checkpoint(path, model, device)` with DDP module prefix stripping.
- **`_cfg_to_dict` passthrough**: `model_builder._cfg_to_dict()` uses `__dict__` traversal instead of manual key whitelist — new model params passed automatically.
- **Config schema normalization**: `_normalize_config_schema()` flattens `data.{split}.loader.*` → `data.{split}.*` keys; `default_runtime.yaml` switched to Schema B flat keys. Backward-compatible.
- **`_cmd_infer` multi-modal rewrite**: Now auto-detects model type: classification (top-5 logits), detection (bbox+scores), video (clip-level prediction). No hardcoded torchvision transforms.

### Removed

- **Dead code**: `components/backbones/simple_cnn.py` (no references anywhere)
- **Duplicated routing**: `engine.py:_build_video_model` + `engine.py:_build_model` if/elif chains (~205 lines)

### Changed

- `cli.py`: `_build_model_from_config` delegates to `model_builder.build_model_from_config()`
- `engine.py`: `_build_model` delegates to `model_builder`
- 11 action recognition `runtime.py` files: `forward_train`/`forward_test` imported from `common_video_forward`
- `config/_base_/default_runtime.yaml`: switched from nested `loader:` to flat Schema B keys
- `config/loader.py`: added `_normalize_config_schema()` call in `load_cv_config()`
- `AGENTS.md`, `docs/status/current.md`, `docs/guides/extension.md`: refreshed for v0.5.0

### Tests

- 29 new tests: `tests/test_model_builder.py` (24 tests: registry completeness + model build smoke) + `tests/test_checkpoint_utils.py` (5 tests: auto-discovery, loading, module-prefix stripping)
- All 140+ regression tests pass

## [0.4.0] — 2026-10-20 · Phase 7: YOLO Integration

### Added — YOLO Detection Models (zero ultralytics dependency)

- **YOLOv10 (n/s/m)**: NMS-free dual head (One2Many + One2One), V10Loss with dual TAL assigners, NMS-free post-processing
- **YOLOv11 (n/s/m/l/x)**: C3k2 backbone + standard decoupled head, CIoU+DFL+BCE loss
- **YOLOv12 (n/s/m/l/x)**: A2C2f attention backbone, PAN+A2C2f neck, CIoU+DFL+BCE loss
- **YOLO26 (n/s/m/l/x)**: Latest YOLO architecture, CIoU+DFL+BCE loss
- **HDMS-YOLO (n/s/m)**: SimSPPF + EMA attention for agricultural weed detection
- **Total**: 21 registered variants, 18 pretrained weight URLs, 22 training config YAMLs

### Shared Components (src/elysia_cv/models/detection/yolo/)

| Category | Files | Lines | Description |
|----------|-------|------:|-------------|
| backbone | `csp_darknet.py`, `a2c2f.py` | 420 | CSPDarkNet + A2C2f attention |
| neck | `pan.py` | 250 | PAN-FPN with version switches |
| head | `detect_head.py`, `v10_head.py` | 500 | Standard + NMS-free dual head |
| losses | `task_aligned_assigner.py`, `v8_loss.py`, `v10_loss.py` | 530 | TAL + CIoU/DFL/BCE + V10 dual |
| postprocess | `nms.py` | 180 | DFL decode + batched NMS |
| infra | `model.py`, `variants.py`, `runtime.py`, `plugin.py`, `weights.py`, `hdms_yolo.py` | 950 | Build, config, training loop, plugin registry |

### Key Features
- **Ultralytics-free weight loading**: StubModule injection via `sys.modules` — loads `.pt` checkpoints without installing `ultralytics`
- **Three-pass key remapping**: Direct → Sequential position → Shape matching
- **Weight cache**: `/data/hdd3/chuqi/pretrained/`
- **HuggingFace cache**: `/data/hdd3/chuqi/cache/huggingface/`
- All 21 variants build + forward + train + grad verified; 140 regression tests pass
- See [docs/plans/yolo_integration.md](docs/plans/yolo_integration.md) for full design

## [0.3.0] — 2026-10-20

### Added — Phase 6: Training Closure

- **EarlyStoppingHook**: patience-based auto-stop with `max`/`min` mode + `min_delta` tolerance; configurable via `eval.early_stop.*` in YAML
- **SWA (Stochastic Weight Averaging)**: periodic snapshot averaging + `apply_shadow`/`restore`/`merge_into`; integrated into EvalHook for automatic eval-time swap; configurable via `engine.optimization.swa_*`
- **TTA (Test-Time Augmentation)**: `tta_predict()` multi-scale + horizontal flip voting for classification; `TTAClassificationWrapper` for model-wrapping with auto-TTA
- **ClassBalancedSampler**: per-class equal-probability sampler for imbalanced datasets; `compute_class_weights()` with effective-number (Cui et al., 2019) and inverse-frequency modes
- **ADE20KDataset**: 150-class semantic segmentation dataset; auto-pairs `*_seg.png` label maps; returns `{"gt_semantic_seg"}` tensor
- **CityscapesDataset**: 19-class urban-scene segmentation; labelId→trainId mapping (7→0 road, etc.); `IGNORE_ID=255`
- **Multi-task loss_weights**: FCOS `forward_train` accepts `loss_weights` dict; `DetectionModel` stores `loss_weights` from config; default `{"cls": 1.0, "reg": 1.0}`
- **tools/profile.py**: FLOPs + parameter count per-layer for any model; supports Conv2d/Linear/BatchNorm/LayerNorm/Pool/Activation; `--verbose` per-layer breakdown
- **CachedDataset**: preload dataset into memory (or LMDB) for fast iteration; `cache_mode="memory"` / `"lmdb"`; hash-based cache validation; works with detection-style list-of-tensor labels

### Changed
- `engine.py`: fix EMA variable ordering bug; add SWA + EarlyStopping construction
- `EvalHook`: accept `swa` param; eval-time weight swap: EMA → SWA; reverse-order restore
- `config/loader.py`: add `loss_weights`, `swa_*`, `early_stop_*` to field map; add `("data",)` to open prefixes
- `config/sections.py`: `_OptSubSection` + `EvalSection` now carry SWA + early_stop fields
- All 6 doc files synced to v0.3.0 + Phase 6 features

## [0.2.0] — 2026-10-20

### Added — Phase 1: Core Loop
- AMP mixed precision + GradScaler in `TrainingEngine`/`TrainerLoop`
- `test`/`infer` CLI: real eval loop with checkpoint auto-discovery
- METRICS registry: `Accuracy`, `TopK`, `MeanAP`, `MeanIoU`
- Cassava classification decoupled from framework core

### Added — Phase 2: Multi-task Support
- FCOS anchor-free detection: center-sampling, true loss, NMS, COCO mAP
- `CocoDetectionDataset`: COCO JSON parsing, variable-size collation
- FCN segmentation: CE/Dice loss, SegHead, MeanIoU
- EvalHook auto-routing: classification ↔ detection ↔ segmentation
- Detection collate_fn for variable-size images

### Added — Phase 3: Developer Experience
- `--resume`/`-r`: full checkpoint restore (model/optimizer/scheduler/step)
- `MessageHub`: global scalar state bus with best-value tracking
- EMA log smoothing in LoggerHook
- DataPrefetcher: CUDA stream async GPU prefetch
- `elysia config dump`: resolve `_base_` inheritance to YAML/JSON
- `DistributedSampler.set_epoch()` per-epoch shuffle
- `Compose.__repr__` pipeline visualization

### Added — Phase 4: Quality Polish
- Rename `faster_rcnn` → `fcos` (27 references across code/docs/tests)
- Detection-specific config validation (head.num_classes, FPN neck compat)
- Data Adapters: HuggingFace datasets, torchvision, VOC detection format
- Optimizers: add Adam + RMSprop (was just SGD + AdamW)
- FSDP: `wrap_fsdp` + `fsdp_state_dict`, engine strategy routing
- Detection visualization: `draw_bboxes` + `draw_segmentation_overlay`

### Added — Phase 5: Training Ecosystem
- **Model EMA**: shadow weights + ramp-up decay + apply_shadow/restore; auto-swap during eval
- **MixUp**: batch-level image blending with Beta-sampled lambda, soft one-hot labels
- **CutMix**: batch-level rectangular region replacement with area-weighted soft labels
- **Mosaic**: 4→1 image stitching for detection, bbox coordinate synchronization
- **RandomRotation**: per-sample rotation ±N° with bbox corner transform
- **RandomAffine**: translation + scale + shear with PIL affine matrix, bbox sync
- **PR curves**: `MeanAP.get_pr_data()` → 101-pt per-class Precision-Recall → `plot_pr_curves()` PNG
- **Training curves**: `MetricHistory` Hook auto-records metrics → `training_curves.png`
- **GT vs Pred visualization**: `draw_gt_pred_comparison()` side-by-side (green GT, red pred)
- **CI/CD**: `.github/workflows/ci.yml` — lint (ruff) + test (py3.10/3.11) + coverage (≥30%)
- **Testing**: 47 new tests → 163 total, 52% code coverage
- `batch_transforms` pipeline in Compose + config schema

### Changed
- All 5 user-facing docs fully updated to v0.2.0 + Phase 5
- Backbone dependency: ResNet/ViT weights via torchvision (no custom weights)

### Known Gaps
- No ONNX/TensorRT/TorchScript export (not a priority)
- No pretrained weight zoo (not a priority)
- COCO full-training validation pending (needs COCO 2017 dataset)