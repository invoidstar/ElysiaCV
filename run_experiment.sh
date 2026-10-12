#!/usr/bin/env bash
# ==========================================================================
# run_experiment.sh — unified training launcher for ElysiaCV
#
# Modeled after ElysiaRobot's run_policy.sh.  One config YAML, one command.
#
# Usage:
#   bash run_experiment.sh configs/classification/cassava_resnet50.yaml
#   bash run_experiment.sh configs/classification/cassava_resnet50.yaml --gpu 6
#   bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --dry-run
#   bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --set optim.params.lr=0.001
# ==========================================================================
set -euo pipefail

# ── Resolve project root ───────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ── Resolve Python ─────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"

# Auto-detect conda python if not in env
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        PYTHON_BIN="$CONDA_PREFIX/bin/python"
    elif [[ -d "/data/hdd3/chuqi/anaconda3/envs/CV" ]]; then
        PYTHON_BIN="/data/hdd3/chuqi/anaconda3/envs/CV/bin/python"
    else
        echo "ERROR: python not found. Activate the 'CV' conda environment first."
        exit 1
    fi
fi

# ── Safe environment ───────────────────────────────────────────────────
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/elysia_mpl}"
mkdir -p "$MPLCONFIGDIR"
export PYTHONDONTWRITEBYTECODE=1

# ── Defaults ───────────────────────────────────────────────────────────
GPU=""
SET_OVERRIDES=()
DRY_RUN=0
PRINT_COMMAND=0

_print_usage() {
    cat << 'USAGE'
Usage: bash run_experiment.sh <config.yaml> [options]

Options:
  --gpu N               Use GPU N (sets CUDA_VISIBLE_DEVICES=N)
  --set KEY=VALUE ...   Override config values (e.g. --set optim.params.lr=0.001)
  --dry-run, -n         Validate config without training
  --print-command       Print the command without executing
  -h, --help            Show this help

Examples:
  bash run_experiment.sh configs/experiments/cassava_resnet50.yaml
  bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --gpu 6
  bash run_experiment.sh configs/experiments/cassava_vit_b16.yaml --gpu 6 --set optim.params.lr=0.0005

Available configs:
USAGE
    find "$PROJECT_ROOT/configs" -name '*.yaml' -not -path '*_base_*' | sort | while read -r f; do
        echo "  ${f#$PROJECT_ROOT/}"
    done
}

# ── Parse positional: config file ──────────────────────────────────────
if [[ $# -eq 0 ]]; then
    _print_usage
    exit 1
fi

# Allow --help / -h without config
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    _print_usage
    exit 0
fi

CONFIG="$1"
shift

# Resolve config path (allow relative to project root)
if [[ ! -f "$CONFIG" ]]; then
    if [[ -f "$PROJECT_ROOT/$CONFIG" ]]; then
        CONFIG="$PROJECT_ROOT/$CONFIG"
    else
        echo "ERROR: config file not found: $CONFIG"
        exit 1
    fi
fi

# ── Parse flags ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --gpu=*)
            GPU="${1#*=}"
            shift
            ;;
        --set)
            # Collect all --set KEY=VALUE pairs
            while [[ $# -gt 1 && "$2" != --* ]]; do
                SET_OVERRIDES+=("$2")
                shift
            done
            shift
            ;;
        --dry-run|-n)
            DRY_RUN=1
            shift
            ;;
        --print-command)
            PRINT_COMMAND=1
            shift
            ;;
        -h|--help)
            _print_usage
            exit 0
            ;;
        *)
            echo "Unknown flag: $1"
            exit 1
            ;;
    esac
done

# ── GPU setup ──────────────────────────────────────────────────────────
if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "[launcher] GPU: $GPU ($(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown'))"
else
    echo "[launcher] GPU: auto (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
fi

# ── Build command ──────────────────────────────────────────────────────
CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/tools/train.py"
    "$CONFIG"
)

# Append overrides
for ov in "${SET_OVERRIDES[@]}"; do
    CMD+=(--set "$ov")
done

# ── Conda activation hint ──────────────────────────────────────────────
CONDA_PREFIX="${CONDA_PREFIX:-}"
if [[ -z "$CONDA_PREFIX" ]] || [[ "$CONDA_PREFIX" != *"CV"* ]]; then
    echo "[launcher] WARNING: 'CV' conda environment not active."
    echo "           Run: conda activate CV"
    echo "           Then re-run this script."
    echo ""
fi

# ── Dry-run mode ───────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[launcher] DRY-RUN MODE — validating config only"
    echo "  Config:   $CONFIG"
    echo "  GPU:      ${GPU:-auto}"
    echo "  Overrides: ${#SET_OVERRIDES[@]}"
    for ov in "${SET_OVERRIDES[@]}"; do
        echo "    --set $ov"
    done
    echo ""
    echo "  Command:"
    echo "    ${CMD[*]}"

    # Actually try loading the config
    echo ""
    "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT/src')
from elysia_cv.config import load_cv_config_with_object
cfg = load_cv_config_with_object('$CONFIG')
print('  run.name:', cfg.run.name)
print('  run.stage:', cfg.run.stage)
print('  model.type:', cfg.model.type)
print('  model.backbone:', cfg.model.backbone)
print('  model.head:', cfg.model.head)
# Override simulation
overrides = '${SET_OVERRIDES[*]}'.split()
if overrides and overrides[0]:
    print(f'  overrides: {overrides}')
print('')
print('  ✅ Config is valid.')
"
    exit 0
fi

# ── Print-command mode ─────────────────────────────────────────────────
if [[ $PRINT_COMMAND -eq 1 ]]; then
    echo "${CMD[*]}"
    exit 0
fi

# ── Run ────────────────────────────────────────────────────────────────
echo "[launcher] Config:   $CONFIG"
echo "[launcher] Overrides: ${#SET_OVERRIDES[@]}"
for ov in "${SET_OVERRIDES[@]}"; do
    echo "           --set $ov"
done
echo ""

exec "${CMD[@]}"