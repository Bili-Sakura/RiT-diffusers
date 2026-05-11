#!/usr/bin/env bash
# Compute per-channel normalization statistics for DINOv2 features on ImageNet.
# Usage: bash scripts/calc_stats.sh [dinov2s|dinov2b]
set -ex

VARIANT=${1:-dinov2s}
DATA_PATH=${DATA_PATH:-imagenet/train}

EXTRA=""
if [ "$VARIANT" = "dinov2s" ]; then
    EXTRA="--dinov2-small"
elif [ "$VARIANT" = "dinov2b" ]; then
    EXTRA=""
else
    echo "Unknown variant: $VARIANT (expected dinov2s or dinov2b)"; exit 1
fi

torchrun --standalone --nproc_per_node=8 calculate_stat.py \
    --data-path ${DATA_PATH} \
    --sample-dir stats/ \
    --rae-model RAE_DINOv2 \
    ${EXTRA}
