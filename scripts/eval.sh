#!/usr/bin/env bash
# Evaluate the released RiT-XL checkpoint on ImageNet 256x256.
# Matches the paper's reported numbers: Heun 25 steps, time-shift schedule.
#   - CFG=1.0:  FID ~1.45
#   - CFG=3.7 (with CFG interval [0.1, 0.98]):  FID ~1.14
#
# On first run, the RiT-XL checkpoint and the matching RAE decoder are
# downloaded automatically from HuggingFace (le723z/RiT and
# nyu-visionx/RAE-collections).
#
# Tip: `bash scripts/eval.sh` does NOT inherit your shell's `conda activate`
# state. Either activate the env first, or pass PYTHON / TORCHRUN explicitly:
#   PYTHON=/path/to/venv/bin/python TORCHRUN=/path/to/venv/bin/torchrun \
#       bash scripts/eval.sh
set -ex

CKPT=${CKPT:-output/rit_xl_dinov2s/checkpoint-last.pth}
OUTPUT_DIR=${OUTPUT_DIR:-output/eval}
IMAGENET_PATH=${IMAGENET_PATH:-imagenet/}
CFG=${CFG:-3.7}
NUM_STEPS=${NUM_STEPS:-25}

# wandb is disabled by default — eval is a one-shot run and wandb.init()
# would otherwise hang on rank 0 while it waits for a login token. To log
# this eval to a wandb project instead, do:
#   export WANDB_MODE=online  WANDB_API_KEY=...  WANDB_PROJECT=RiT
export WANDB_MODE=${WANDB_MODE:-disabled}

# Resolve interpreters: prefer the active venv's python/torchrun, else fall
# back to whatever is on PATH. If PYTHON is unset but TORCHRUN was resolved,
# derive PYTHON from the same bin/ — this avoids the common case where
# `torchrun` lives in a conda env that has torch but the default `python`
# points elsewhere.
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

# Auto-download RiT-XL checkpoint + RAE decoder if missing.
"${PYTHON}" scripts/download_assets.py --assets rae_decoder_small rit_xl_ckpt

"${TORCHRUN}" --nproc_per_node=8 main.py \
    --model RiT-XL/16 \
    --rae_model RAE_DINOv2 --dinov2small \
    --rae_normalize \
    --normalization_stat_path stats/RAE_DINOv2_small/normalization_stats.pt \
    --use_cls --cls_loss_weight 0.2 \
    --pred_type x \
    --P_mean -0.8 --P_std 0.8 \
    --img_size 256 --noise_scale 1.0 \
    --gen_bsz 4096 --num_images 50000 \
    --num_sampling_steps ${NUM_STEPS} --sampling_method heun \
    --sample_schedule time_shift \
    --cfg ${CFG} --interval_min 0.1 --interval_max 0.98 \
    --gen_precision fp32 \
    --coupled_noise \
    --checkpoint_path ${CKPT} \
    --output_dir ${OUTPUT_DIR} \
    --data_path ${IMAGENET_PATH} --evaluate_gen
