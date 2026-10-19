#!/usr/bin/env bash
# ==========================================================================
# Cassava Fine-Grained Classification — Experiment Comparison
#
# Usage:
#   bash scripts/run_cassava_experiments.sh              # 单卡 (GPU 6)
#   bash scripts/run_cassava_experiments.sh --gpu 7      # 指定 GPU
#   bash scripts/run_cassava_experiments.sh --ddp 2      # 2 卡 DDP
#   bash scripts/run_cassava_experiments.sh --gpu 4 --ddp 2  # GPU 4,5
# ==========================================================================
set -euo pipefail

GPU=6
DDP_NPROC=0

# ── Parse top-level flags ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2";          shift 2 ;;
        --gpu=*)      GPU="${1#*=}";     shift ;;
        --ddp)        DDP_NPROC="$2";    shift 2 ;;
        --ddp=*)      DDP_NPROC="${1#*=}"; shift ;;
        -h|--help)
            echo "Usage: bash scripts/run_cassava_experiments.sh [--gpu N] [--ddp N]"
            exit 0
            ;;
        *) shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs/experiments"

# ── Header ─────────────────────────────────────────────────────────────
echo "=============================================="
echo " Cassava Fine-Grained Experiment Suite"
echo "=============================================="
if [[ $DDP_NPROC -gt 0 ]]; then
    echo " Mode:          DDP ($DDP_NPROC GPUs)"
    echo " GPU:           ${GPU} → $((GPU + DDP_NPROC - 1))"
else
    echo " GPU:           $GPU ($(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown))"
fi
echo " Epochs:        30"
echo " Eff. Batch:    32"
echo " Output:        outputs/checkpoints/cassava_<model>/"
echo "=============================================="
echo ""

# ── Experiment definitions ─────────────────────────────────────────────
# Format: name|config_file|extra_overrides

EXPERIMENTS=(
    # name              config_file               extra overrides
    "mlp               cassava_mlp.yaml"
    "simple_cnn        cassava_simple_cnn.yaml"
    "resnet18          cassava_resnet18.yaml"
    "resnet34          cassava_resnet34.yaml"
    "resnet50          cassava_resnet50.yaml"
    "vit_b16           cassava_vit_b16.yaml"
    "vit_l16           cassava_vit_l16.yaml"
)

# ── Run experiments sequentially ───────────────────────────────────────
run_experiment() {
    local name="$1"
    local config="$2"
    shift 2
    local overrides=("$@")

    local output_dir="$PROJECT_DIR/outputs/checkpoints/cassava_${name}"
    mkdir -p "$output_dir"
    local log_file="$output_dir/train.log"
    local config_path="$CONFIG_DIR/$config"

    echo ""
    echo "----------------------------------------"
    echo "  [$name] Starting..."
    echo "  Config:  $config"
    echo "  Log:     $log_file"
    echo "----------------------------------------"

    # Build override args
    local set_args=(--set "run.output_dir=./outputs/checkpoints/cassava_${name}")
    for ov in "${overrides[@]}"; do
        [[ -n "$ov" ]] && set_args+=(--set "$ov")
    done

    # Build launcher args
    local launcher_args=("$config_path")
    if [[ $DDP_NPROC -gt 0 ]]; then
        launcher_args+=(--ddp "$DDP_NPROC" --gpu "$GPU")
    else
        launcher_args+=(--gpu "$GPU")
    fi
    launcher_args+=("${set_args[@]}")

    # Use run_experiment.sh for unified launch
    bash "$PROJECT_DIR/run_experiment.sh" \
        "${launcher_args[@]}" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "  [$name] FAILED (exit code=$exit_code)"
        echo "  See log: $log_file"
        return $exit_code
    fi

    echo "  [$name] COMPLETE"
}

# ── Main ───────────────────────────────────────────────────────────────
FAILED=()
for exp_line in "${EXPERIMENTS[@]}"; do
    IFS=' ' read -ra parts <<< "$exp_line"
    name="${parts[0]}"
    config="${parts[1]}"
    overrides=("${parts[@]:2}")

    if ! run_experiment "$name" "$config" "${overrides[@]}"; then
        FAILED+=("$name")
    fi
done

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Experiment Suite Complete"
echo "=============================================="
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED experiments:"
    for f in "${FAILED[@]}"; do
        echo "  - $f → outputs/checkpoints/cassava_${f}/train.log"
    done
    exit 1
else
    echo "All ${#EXPERIMENTS[@]} experiments completed successfully!"
    echo ""
    echo "Results (outputs/checkpoints/):"
    for exp_line in "${EXPERIMENTS[@]}"; do
        read -ra parts <<< "$exp_line"
        name="${parts[0]}"
        echo "  cassava_${name}/  ├── train.log"
        echo "                    ├── checkpoint_best_Accuracy.pt"
        echo "                    └── checkpoint_final.pt"
    done
fi