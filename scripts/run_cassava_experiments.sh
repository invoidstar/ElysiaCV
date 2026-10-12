#!/usr/bin/env bash
# ==========================================================================
# Cassava Fine-Grained Classification — Experiment Comparison
#
# GPU:        6 (RTX 3090, 24GB)
# Epochs:     30  (all models)
# Eff. Batch: 32  (all models, via gradient accumulation for ViT)
#
# ViT-B: batch=16 + grad_accum=2 → effective batch=32
# ViT-L: batch=8  + grad_accum=4 → effective batch=32
# Others: batch=32
# ==========================================================================
set -euo pipefail

GPU=6
export CUDA_VISIBLE_DEVICES=$GPU

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs/experiments"

echo "=============================================="
echo " Cassava Fine-Grained Experiment Suite"
echo "=============================================="
echo " GPU:           $GPU ($(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown))"
echo " Epochs:        30"
echo " Eff. Batch:    32"
echo " Output:        work_dirs/cassava_<model>/"
echo "=============================================="
echo ""

# ── Experiment definitions ─────────────────────────────────────────────
# Format: name|config_file|extra_overrides (space-separated key=value pairs)

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

    local output_dir="$PROJECT_DIR/work_dirs/cassava_${name}"
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
    local set_args=(--set "run.output_dir=./work_dirs/cassava_${name}")
    for ov in "${overrides[@]}"; do
        [[ -n "$ov" ]] && set_args+=(--set "$ov")
    done

    # Use run_experiment.sh for unified launch
    bash "$PROJECT_DIR/run_experiment.sh" \
        "$config_path" \
        --gpu "$GPU" \
        "${set_args[@]}" \
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
        echo "  - $f → work_dirs/cassava_${f}/train.log"
    done
    exit 1
else
    echo "All ${#EXPERIMENTS[@]} experiments completed successfully!"
    echo ""
    echo "Results (work_dirs/):"
    for exp_line in "${EXPERIMENTS[@]}"; do
        read -ra parts <<< "$exp_line"
        name="${parts[0]}"
        echo "  cassava_${name}/  ├── train.log"
        echo "                    ├── checkpoint_best.pt"
        echo "                    └── checkpoint_final.pt"
    done
fi