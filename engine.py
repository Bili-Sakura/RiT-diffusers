"""Training and evaluation engine for RiT.

`train_one_epoch` runs a single distributed training epoch over the frozen
RAE encoder + trainable Denoiser. `evaluate` generates 50K images, writes them
to a shared directory, and computes FID / IS / Precision / Recall on rank 0
via torch-fidelity.
"""
import copy
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_fidelity
import wandb

import util.lr_sched as lr_sched
import util.misc as misc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, model_without_ddp, rae_model, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    optimizer.zero_grad()

    if log_writer:
        print(f'log_dir: {args.output_dir}')

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        progress = data_iter_step / len(data_loader) + epoch

        lr_sched.adjust_learning_rate(optimizer, progress, args)

        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)

        x, cls_token = rae_model.encode(x, return_cls_token=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, loss_dict = model(x, labels, cls_token=cls_token)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        metric_logger.update(grad_norm=grad_norm)
        for k, v in loss_dict.items():
            if k != "loss_total":
                metric_logger.update(**{k: v.item()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if data_iter_step % args.log_freq == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            grad_norm_reduce  = misc.all_reduce_mean(grad_norm)
            if log_writer and misc.is_main_process():
                epoch_1000x = int(progress * 1000)
                log_data = {
                    'lr': lr,
                    'grad_norm': grad_norm_reduce,
                    'train_loss': loss_value_reduce,
                    'epoch_1000x': epoch_1000x,
                }
                for k, v in loss_dict_reduced.items():
                    if "loss" in k:
                        log_data[f'train_{k}'] = v.item()
                wandb.log(log_data)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_without_ddp, rae_model, args, epoch, batch_size=64, log_writer=None):
    """Generate `args.num_images` samples and compute FID/IS/Precision/Recall.

    All ranks write images to the same shared folder; rank 0 then calls
    torch-fidelity to compute the metrics against a pre-computed reference.
    """
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps  = args.num_images // (batch_size * world_size) + 1

    suffix = "{}-steps{}-cfg{}-cfgcls{}-interval{}-{}-clsinterval{}-{}-cn{}-image{}-res{}".format(
        model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
        model_without_ddp.cfg_cls_scale,
        model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1],
        model_without_ddp.cfg_cls_interval[0], model_without_ddp.cfg_cls_interval[1],
        int(model_without_ddp.coupled_noise),
        args.num_images, args.img_size,
    )
    save_folder = args.save_image_dir or os.path.join(args.output_dir, "sampled_images", suffix)

    print("Save to:", save_folder)
    if local_rank == 0:
        os.makedirs(save_folder, exist_ok=True)
    torch.distributed.barrier()

    existing_images = len(os.listdir(save_folder)) if os.path.isdir(save_folder) else 0
    skip_generation = existing_images >= args.num_images
    if skip_generation:
        print(f"Found {existing_images} images (>= {args.num_images}), skipping generation.")
    else:
        print(f"Found {existing_images} images (< {args.num_images}), generating...")

    if not skip_generation:
        # Swap to EMA params for sampling, restore after.
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict   = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = model_without_ddp.ema_params1[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

        class_num = args.class_num
        assert args.num_images % class_num == 0
        class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
        class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

        for i in range(num_steps):
            print(f"Generation step {i}/{num_steps}")

            start_idx  = world_size * batch_size * i + local_rank * batch_size
            labels_gen = class_label_gen_world[start_idx: start_idx + batch_size]
            labels_gen = torch.Tensor(labels_gen).long().cuda()

            gen_dtype = torch.bfloat16 if args.gen_precision == 'bf16' else None
            with torch.amp.autocast('cuda', enabled=gen_dtype is not None, dtype=gen_dtype or torch.float32):
                sampled_images = model_without_ddp.generate(labels_gen)

            sampled_images = rae_model.decode(sampled_images).detach()

            batch_global_start = i * batch_size * world_size + local_rank * batch_size
            n_valid = min(batch_size, max(0, args.num_images - batch_global_start))
            if n_valid <= 0:
                continue

            valid_images = sampled_images[:n_valid]
            imgs_cpu = valid_images.cpu()
            for b_id in range(n_valid):
                img_id  = batch_global_start + b_id
                gen_img = np.round(np.clip(imgs_cpu[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
                gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_folder, f'{str(img_id).zfill(5)}.png'), gen_img)

        torch.distributed.barrier()
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)
    else:
        torch.distributed.barrier()

    if args.img_size == 256:
        fid_statistics_file = 'fid_stats/imagenet256_stats.npz'
    elif args.img_size == 512:
        fid_statistics_file = 'fid_stats/imagenet512_stats.npz'
    else:
        raise NotImplementedError

    # FID / IS / Precision / Recall: only rank 0 computes (serial). `log_writer`
    # controls wandb logging only — it does NOT gate the metrics themselves.
    wandb_on = bool(log_writer)
    if local_rank == 0:
        if wandb_on:
            image_files = sorted(os.listdir(save_folder))[:50]
            print(f"[eval] save_folder={save_folder}, total files={len(os.listdir(save_folder))}, wandb sample={len(image_files)}")
            wandb_images = []
            for img_file in image_files:
                img_path = os.path.join(save_folder, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[eval] WARNING: cv2.imread failed for {img_path}, skipping")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                wandb_images.append(wandb.Image(img, caption=img_file))
            wandb.log({"generated_images": wandb_images, "epoch": epoch})

        print("Calculating FID and Inception Score...")
        n_images = len(os.listdir(save_folder))
        print(f"Computing FID/IS with {n_images:,} images")
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True, isc=True, fid=True, kid=False, prc=False, verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']

        print("Computing Precision/Recall...")
        ref_npz_url = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
        ref_cache_dir  = os.path.join(os.path.dirname(args.output_dir), '_ref_cache')
        ref_npz_path   = os.path.join(ref_cache_dir, 'VIRTUAL_imagenet256_labeled.npz')
        ref_images_dir = os.path.join(ref_cache_dir, 'ref_images')

        if not os.path.exists(ref_npz_path):
            print(f"Downloading reference npz to {ref_npz_path}...")
            import urllib.request
            os.makedirs(os.path.dirname(ref_npz_path), exist_ok=True)
            urllib.request.urlretrieve(ref_npz_url, ref_npz_path)
            print("Download complete.")

        os.makedirs(ref_images_dir, exist_ok=True)
        if len(os.listdir(ref_images_dir)) == 0:
            print(f"Unpacking {ref_npz_path} to {ref_images_dir}...")
            data = np.load(ref_npz_path)
            arr = data['arr_0']
            for idx in range(arr.shape[0]):
                img_bgr = arr[idx][:, :, ::-1]
                cv2.imwrite(os.path.join(ref_images_dir, f'{idx:05d}.png'), img_bgr)
            print(f"Unpacked {arr.shape[0]} images.")
        else:
            print(f"Reference images already unpacked ({len(os.listdir(ref_images_dir))} files).")
        prc_dict = torch_fidelity.calculate_metrics(
            input1=ref_images_dir,
            input2=save_folder,
            cuda=True, isc=False, fid=False, kid=False, prc=True, verbose=False,
        )
        precision = prc_dict['precision']
        recall    = prc_dict['recall']
        _log_metrics(fid, inception_score, precision, recall, model_without_ddp, args, epoch, wandb_on)

    torch.distributed.barrier()
    if local_rank == 0:
        shutil.rmtree(save_folder, ignore_errors=True)
    torch.distributed.barrier()


def _log_metrics(fid, inception_score, precision, recall, model_without_ddp, args, epoch, wandb_on):
    postfix = f"_cfg{model_without_ddp.cfg_scale}_res{args.img_size}"
    if wandb_on:
        wandb.log({
            f'fid{postfix}': fid,
            f'is{postfix}': inception_score,
            f'precision{postfix}': precision,
            f'recall{postfix}': recall,
            'epoch': epoch,
        })
    print(f"FID: {fid:.4f}, Inception Score: {inception_score:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result_line = (
        f"Experiment: {Path(args.output_dir).name}, "
        f"Epoch: {epoch}, "
        f"FID: {fid:.4f}, IS: {inception_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"cfg: {model_without_ddp.cfg_scale}, cfg_cls: {model_without_ddp.cfg_cls_scale}, "
        f"interval: {model_without_ddp.cfg_interval[0]}-{model_without_ddp.cfg_interval[1]}, "
        f"cls_interval: {model_without_ddp.cfg_cls_interval[0]}-{model_without_ddp.cfg_cls_interval[1]}, "
        f"coupled_noise: {int(model_without_ddp.coupled_noise)}, "
        f"sampler: {model_without_ddp.method}, steps: {model_without_ddp.steps}, "
        f"sample_schedule: {model_without_ddp.sample_schedule}, "
        f"precision: {args.gen_precision}, "
        f"num_images: {args.num_images}, res: {args.img_size}, "
        f"Time: {now_str}\n"
    )

    result_save_dir = Path(args.output_dir)
    result_save_dir.mkdir(parents=True, exist_ok=True)
    output_txt_file = result_save_dir / "fid_results.txt"
    with output_txt_file.open('a') as f:
        f.write(result_line)
    print(f"Results appended to {output_txt_file}")

    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open('a') as f:
            f.write(result_line)
        print(f"Results appended to {summary_path}")
