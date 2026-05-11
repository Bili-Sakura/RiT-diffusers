#!/usr/bin/env bash
# Evaluate a trained RiT-XL checkpoint on ImageNet 256x256.
# Matches the paper's reported numbers: Heun 25 steps, time-shift schedule.
#   - CFG=1.0:  FID ~1.45
#   - CFG=3.7 (with CFG interval [0.1, 0.98]):  FID ~1.14
set -ex

CKPT=${CKPT:-output/rit_xl_dinov2s/checkpoint-last.pth}
OUTPUT_DIR=${OUTPUT_DIR:-output/eval}
IMAGENET_PATH=${IMAGENET_PATH:-imagenet/}
CFG=${CFG:-3.7}
NUM_STEPS=${NUM_STEPS:-25}

torchrun --nproc_per_node=8 main.py \
    --model RiT-XL/16 \
    --rae_model RAE_DINOv2 --dinov2small \
    --rae_normalize \
    --normalization_stat_path stats/RAE_DINOv2_small/normalization_stats.pt \
    --reg_loss --reg_loss_weight 0.2 \
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
