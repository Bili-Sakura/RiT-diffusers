#!/usr/bin/env bash
# Ablation: original JiT logit-normal noise schedule (no dimension-aware time shift).
# Reproduces the "Logit-normal" row of Table 3 (~3.17 FID at 800 ep).
set -ex

OUTPUT_DIR=${OUTPUT_DIR:-output/ablation_logit_normal}
IMAGENET_PATH=${IMAGENET_PATH:-imagenet/}

torchrun --nproc_per_node=8 main.py \
    --model RiT-XL/16 \
    --rae_model RAE_DINOv2 --dinov2small \
    --rae_normalize \
    --normalization_stat_path stats/RAE_DINOv2_small/normalization_stats.pt \
    --reg_loss --reg_loss_weight 0.2 \
    --pred_type x \
    --time_schedule logit_normal \
    --P_mean 0.0 --P_std 1.0 \
    --img_size 256 --noise_scale 1.0 \
    --batch_size 192 --blr 5e-5 \
    --epochs 800 --warmup_epochs 5 \
    --gen_bsz 4096 --num_images 50000 \
    --num_sampling_steps 50 --sampling_method heun --sample_schedule logit_normal \
    --cfg 1.0 --interval_min 0.1 --interval_max 1.0 \
    --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
    --data_path ${IMAGENET_PATH} --online_eval
