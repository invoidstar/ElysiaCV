#!/usr/bin/env bash
# ==========================================================================
# run_experiment.sh — unified training launcher for ElysiaCV
#
# Modeled after ElysiaRobot's run_policy.sh.  One config YAML, one command.
#
# Usage:
#   bash run_experiment.sh configs/experiments/cassava_resnet50.yaml
#   bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --gpu 6
#   bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --ddp 2
#   bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --gpu 0,1 --ddp 2
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
DDP_NPROC=0
DDP_MASTER_PORT="${ELYSIA_DDP_PORT:-29500}"
SET_OVERRIDES=()
DRY_RUN=0
PRINT_COMMAND=0

_print_usage() {
    cat << 'USAGE'
Usage: bash run_experiment.sh <config.yaml> [options]

Options:
  --gpu N|N,N,N        GPU device(s).  Single number or comma-separated list.
                         DDP mode auto-expands: --gpu 4 --ddp 2 → GPUs 4,5.
  --ddp N              Enable DDP with N GPUs (via torchrun).
  --master-port PORT   torchrun master port (default: 29500).
  --set KEY=VALUE ...  Override config values (e.g. --set optim.params.lr=0.001)
  --dry-run, -n        Validate config without training.
  --print-command      Print the command without executing.
  -h, --help           Show this help.

Examples:
  # Single GPU
  bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --gpu 6

  # DDP on 2 GPUs (auto-detect from CUDA_VISIBLE_DEVICES)
  bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --ddp 2

  # DDP on specific GPUs (4,5)
  bash run_experiment.sh configs/experiments/cassava_resnet50.yaml --gpu 4 --ddp 2

  # DDP on explicit GPU list
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_experiment.sh configs/experiments/cassava_vit_b16.yaml --ddp 4

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
        --ddp)
            DDP_NPROC="$2"
            shift 2
            ;;
        --ddp=*)
            DDP_NPROC="${1#*=}"
            shift
            ;;
        --master-port)
            DDP_MASTER_PORT="$2"
            shift 2
            ;;
        --master-port=*)
            DDP_MASTER_PORT="${1#*=}"
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
_auto_detect_gpus() {
    nvidia-smi -L 2>/dev/null | wc -l || echo "0"
}

if [[ $DDP_NPROC -gt 0 ]]; then
    # ── DDP mode ───────────────────────────────────────────────────────
    if ! command -v torchrun &>/dev/null; then
        echo "ERROR: torchrun not found. Install PyTorch >= 1.10 or set PATH."
        exit 1
    fi

    if [[ -z "$GPU" ]]; then
        # Auto-detect: use CUDA_VISIBLE_DEVICES if set, else all GPUs
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            IFS=',' read -ra gpu_list <<< "$CUDA_VISIBLE_DEVICES"
        else
            total=$(_auto_detect_gpus)
            gpu_list=($(seq 0 $((total - 1))))
        fi

        if [[ ${#gpu_list[@]} -lt $DDP_NPROC ]]; then
            echo "ERROR: need $DDP_NPROC GPUs for DDP, but only ${#gpu_list[@]} available."
            echo "       Set CUDA_VISIBLE_DEVICES or use --gpu START_GPU"
            exit 1
        fi

        # Use first N GPUs
        ddp_gpus=("${gpu_list[@]:0:$DDP_NPROC}")
        GPU_STR=$(IFS=','; echo "${ddp_gpus[*]}")
    else
        # --gpu with DDP: expand from starting GPU
        if [[ "$GPU" =~ ^[0-9]+$ ]]; then
            # Single number → range start,start+1,...,start+N-1
            ddp_gpus=($(seq "$GPU" $((GPU + DDP_NPROC - 1))))
            GPU_STR=$(IFS=','; echo "${ddp_gpus[*]}")
        else
            # Already comma-separated list
            IFS=',' read -ra ddp_gpus <<< "$GPU"
            if [[ ${#ddp_gpus[@]} -lt $DDP_NPROC ]]; then
                echo "ERROR: DDP needs $DDP_NPROC GPUs, but only ${#ddp_gpus[@]} given."
                exit 1
            fi
            ddp_gpus=("${ddp_gpus[@]:0:$DDP_NPROC}")
            GPU_STR=$(IFS=','; echo "${ddp_gpus[*]}")
        fi
    fi

    export CUDA_VISIBLE_DEVICES="$GPU_STR"
    echo "[launcher] DDP mode: $DDP_NPROC GPUs ($GPU_STR)"

else
    # ── Single GPU mode ─────────────────────────────────────────────────
    if [[ -z "$GPU" ]]; then
        GPU="${CUDA_VISIBLE_DEVICES:-auto}"
        echo "[launcher] GPU: $GPU"
    else
        export CUDA_VISIBLE_DEVICES="$GPU"
        echo "[launcher] GPU: $GPU ($(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown'))"
    fi
fi

# ── Build command ──────────────────────────────────────────────────────
if [[ $DDP_NPROC -gt 0 ]]; then
    CMD=(
        torchrun
        --nproc_per_node="$DDP_NPROC"
        --master_port="$DDP_MASTER_PORT"
        "$PROJECT_ROOT/tools/train.py"
        "$CONFIG"
    )
else
    CMD=(
        "$PYTHON_BIN" "$PROJECT_ROOT/tools/train.py"
        "$CONFIG"
    )
fi

# Append overrides (single --set with all values, argparse nargs="*" semantics)
if [ ${#SET_OVERRIDES[@]} -gt 0 ]; then
    CMD+=(--set "${SET_OVERRIDES[@]}")
fi

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
    echo "  DDP:      $([[ $DDP_NPROC -gt 0 ]] && echo "yes ($DDP_NPROC GPUs)" || echo "no")"
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