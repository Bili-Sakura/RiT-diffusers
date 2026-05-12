#!/usr/bin/env bash
# Main RiT-XL training recipe: DINOv2-Small encoder + x-prediction +
# element-wise standardization + dimension-aware time shift + joint [CLS]-patch.
# Reproduces the paper's FID 1.45 (no CFG) / 1.14 (w/ CFG=3.7).
#
# `bash scripts/train.sh` does NOT inherit `conda activate`. Either activate
# the env first, or pass interpreters explicitly:
#   PYTHON=/path/to/venv/bin/python TORCHRUN=/path/to/venv/bin/torchrun \
#       bash scripts/train.sh
set -ex

OUTPUT_DIR=${OUTPUT_DIR:-output/rit_xl_dinov2s}
IMAGENET_PATH=${IMAGENET_PATH:-imagenet/}

# wandb is disabled by default so the script runs out of the box without any
# account setup. For long training runs you probably want it enabled:
#   export WANDB_MODE=online  WANDB_API_KEY=...  WANDB_PROJECT=RiT
export WANDB_MODE=${WANDB_MODE:-disabled}

TORCHRUN=${TORCHRUN:-$(command -v torchrun || echo torchrun)}
if [ -z "${PYTHON}" ]; then
    candidate="$(dirname "${TORCHRUN}")/python"
    if [ -x "${candidate}" ]; then
        PYTHON="${candidate}"
    else
        PYTHON=$(command -v python || command -v python3)
    fi
fi
echo "Using PYTHON=${PYTHON}"
echo "Using TORCHRUN=${TORCHRUN}"

# Auto-download the RAE decoder weights (used by both training and online eval).
"${PYTHON}" scripts/download_assets.py --assets rae_decoder_small

"${TORCHRUN}" --nproc_per_node=8 --nnodes=1 main.py \
    --model RiT-XL/16 \
    --rae_model RAE_DINOv2 --dinov2small \
    --rae_normalize \
    --normalization_stat_path stats/RAE_DINOv2_small/normalization_stats.pt \
    --use_cls --cls_loss_weight 0.2 \
    --pred_type x \
    --time_schedule shift \
    --P_mean -0.8 --P_std 0.8 \
    --img_size 256 --noise_scale 1.0 \
    --batch_size 192 --blr 5e-5 \
    --epochs 800 --warmup_epochs 5 \
    --gen_bsz 4096 --num_images 50000 \
    --num_sampling_steps 25 --sampling_method heun --sample_schedule time_shift \
    --cfg 1.0 --interval_min 0.1 --interval_max 1.0 \
    --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
    --data_path ${IMAGENET_PATH} --online_eval
