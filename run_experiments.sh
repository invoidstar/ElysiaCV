#!/bin/bash
# ElysiaCV CropAndWeed YOLO Experiments — sequential on GPU 7
# 26 experiments: 2-class(10) + 9-class(8) + 84-class(8)
# All use same budget: 50 epochs, AdamW lr=0.001, batch=16, 640×640, seed=42

set -euo pipefail

GPU=7
export CUDA_VISIBLE_DEVICES=$GPU
CONDA_ENV="CV"
BASE_CONFIG="configs/detection/_base_/cropweed_base.yaml"
LOG_DIR="./experiment_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── helpers ──
log()   { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }
header(){ echo ""; echo "══════════════════════════════════════════════════════"; echo "$*"; echo "══════════════════════════════════════════════════════"; }

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

log "Starting 26-experiment CropAndWeed YOLO benchmark on GPU $GPU"
log "Log: $MAIN_LOG"

# Activate conda
source /data/hdd3/chuqi/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH=src

# ═══════════════════════════════════════════════════════════════
#  Group 1: 2-class (crop_or_weed2) — 10 experiments
# ═══════════════════════════════════════════════════════════════
header "GROUP 1: 2-class (crop vs weed)"
LABEL=crop_or_weed2
NCLASS=2

for variant in yolov11n yolov11s yolov10n yolov10s yolov12n yolov12s yolov26n yolov26s hdms_yolov11n hdms_yolov11s; do
    NAME="${variant}_2cls"
    OUTDIR="./outputs/checkpoints/2cls/${variant}"

    header "  ⇒ ${variant} (2-class)"
    log "    variant=${variant}  classes=${NCLASS}  label=${LABEL}"

    python tools/train.py "$BASE_CONFIG" \
        --set model.variant=${variant} \
             model.num_classes=${NCLASS} \
             data.train.label_mode=${LABEL} \
             data.val.label_mode=${LABEL} \
             run.name=${NAME} \
             run.output_dir=${OUTDIR} \
             2>&1 | tee -a "$MAIN_LOG"

    log "    ✅ ${variant} done"
done

# ═══════════════════════════════════════════════════════════════
#  Group 2: 9-class (crop_or_weed9) — 8 experiments (skip HDMS)
# ═══════════════════════════════════════════════════════════════
header "GROUP 2: 9-class"

LABEL=crop_or_weed9
NCLASS=9

for variant in yolov11n yolov11s yolov10n yolov10s yolov12n yolov12s yolov26n yolov26s; do
    NAME="${variant}_9cls"
    OUTDIR="./outputs/checkpoints/9cls/${variant}"

    header "  ⇒ ${variant} (9-class)"
    log "    variant=${variant}  classes=${NCLASS}  label=${LABEL}"

    python tools/train.py "$BASE_CONFIG" \
        --set model.variant=${variant} \
             model.num_classes=${NCLASS} \
             data.train.label_mode=${LABEL} \
             data.val.label_mode=${LABEL} \
             run.name=${NAME} \
             run.output_dir=${OUTDIR} \
             2>&1 | tee -a "$MAIN_LOG"

    log "    ✅ ${variant} done"
done

# ═══════════════════════════════════════════════════════════════
#  Group 3: 84-class (full) — 8 experiments (skip HDMS)
# ═══════════════════════════════════════════════════════════════
header "GROUP 3: 84-class (full)"

LABEL=full
NCLASS=84

for variant in yolov11n yolov11s yolov10n yolov10s yolov12n yolov12s yolov26n yolov26s; do
    NAME="${variant}_84cls"
    OUTDIR="./outputs/checkpoints/84cls/${variant}"

    header "  ⇒ ${variant} (84-class)"
    log "    variant=${variant}  classes=${NCLASS}  label=${LABEL}"

    python tools/train.py "$BASE_CONFIG" \
        --set model.variant=${variant} \
             model.num_classes=${NCLASS} \
             data.train.label_mode=${LABEL} \
             data.val.label_mode=${LABEL} \
             run.name=${NAME} \
             run.output_dir=${OUTDIR} \
             2>&1 | tee -a "$MAIN_LOG"

    log "    ✅ ${variant} done"
done

# ═══════════════════════════════════════════════════════════════
header "ALL 26 EXPERIMENTS COMPLETE 🎉"
log "Log saved to: $MAIN_LOG"