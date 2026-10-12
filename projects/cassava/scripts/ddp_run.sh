#!/usr/bin/env bash
# =============================================================================
#  ElysiaCV DDP Launcher — single-model multi-GPU training via torchrun
# =============================================================================
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash projects/cassava/scripts/ddp_run.sh \
#       configs/experiments/cassava_resnet50.yaml
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash projects/cassava/scripts/ddp_run.sh \
#       configs/experiments/cassava_vit_b16.yaml --set schedule.steps=8000
# =============================================================================

set -euo pipefail

CONFIG="${1:?Usage: $0 <config_path> [--set key=value ...]}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Detect GPU count from CUDA_VISIBLE_DEVICES
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NPROC=${#GPU_LIST[@]}
fi

if [ "$NPROC" -lt 2 ]; then
    echo "WARNING: fewer than 2 GPUs detected (nproc=$NPROC). Use single-GPU instead:"
    echo "  python tools/train.py $CONFIG $*"
    exit 1
fi

echo "=============================================="
echo " ElysiaCV DDP Training"
echo "=============================================="
echo " Config:       $CONFIG"
echo " GPUs:         $NPROC ($CUDA_VISIBLE_DEVICES)"
echo " Extra args:   $*"
echo "=============================================="
echo ""

cd "$PROJECT_ROOT"

torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="${ELYSIA_DDP_PORT:-29500}" \
    tools/train.py "$CONFIG" "$@"