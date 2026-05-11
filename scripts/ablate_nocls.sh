#!/usr/bin/env bash
# Ablation: drop joint [CLS]-patch modeling. Reproduces "w/o CLS" row of Table 3.
set -ex

OUTPUT_DIR=${OUTPUT_DIR:-output/ablation_nocls}
IMAGENET_PATH=${IMAGENET_PATH:-imagenet/}

torchrun --nproc_per_node=8 main.py \
    --model RiT-XL/16 \
    --rae_model RAE_DINOv2 --dinov2small \
    --rae_normalize \
    --normalization_stat_path stats/RAE_DINOv2_small/normalization_stats.pt \
    --pred_type x \
    --time_schedule shift \
    --P_mean -0.8 --P_std 0.8 \
    --img_size 256 --noise_scale 1.0 \
    --batch_size 192 --blr 5e-5 \
    --epochs 800 --warmup_epochs 5 \
    --gen_bsz 4096 --num_images 50000 \
    --num_sampling_steps 50 --sampling_method heun --sample_schedule time_shift \
    --cfg 1.0 --interval_min 0.1 --interval_max 1.0 \
    --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
    --data_path ${IMAGENET_PATH} --online_eval
