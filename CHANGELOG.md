# Changelog

All notable changes to ElysiaCV will be documented in this file.

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