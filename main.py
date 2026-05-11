"""Entry point for training and evaluating RiT on ImageNet.

Usage examples:

    # Train RiT-XL on DINOv2-S features (main paper config)
    torchrun --nproc_per_node=8 main.py \\
        --model RiT-XL/16 --rae_model RAE_DINOv2 --dinov2small \\
        --rae_normalize \\
        --normalization_stat_path stats/RAE_DINOv2_small/normalization_stats.pt \\
        --reg_loss --reg_loss_weight 0.2 \\
        --pred_type x --P_mean -0.8 --P_std 0.8 \\
        --batch_size 192 --blr 5e-5 --epochs 800 --warmup_epochs 5 \\
        --output_dir output/rit_xl_dinov2s --data_path imagenet/ --online_eval

    # Evaluate a checkpoint
    torchrun --nproc_per_node=8 main.py \\
        --model RiT-XL/16 --rae_model RAE_DINOv2 --dinov2small --rae_normalize \\
        --normalization_stat_path stats/RAE_DINOv2_small/normalization_stats.pt \\
        --reg_loss --reg_loss_weight 0.2 --pred_type x \\
        --gen_bsz 4096 --num_images 50000 \\
        --num_sampling_steps 25 --sampling_method heun --sample_schedule time_shift \\
        --cfg 3.7 --interval_min 0.1 --interval_max 0.98 \\
        --checkpoint_path path/to/checkpoint.pth \\
        --output_dir output/eval --data_path imagenet/ --evaluate_gen
"""
import argparse
import copy
import datetime
import json
import os
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb

import util.misc as misc
from denoiser import Denoiser
from engine import evaluate, train_one_epoch
from rae import RAE_models
from util.crop import center_crop_arr
from util.muon import split_params_for_muon


def get_args_parser():
    parser = argparse.ArgumentParser('RiT', add_help=False)

    # ---- architecture ----
    parser.add_argument('--model', default='RiT-XL/16', type=str, metavar='MODEL',
                        help='Model variant (see RiT_models registry in model.py)')
    parser.add_argument('--img_size', default=256, type=int, help='Input image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)
    parser.add_argument('--pred_type', default='x', choices=['x', 'v'],
                        help='Network output parameterization (paper default: x-prediction)')

    # ---- RAE encoder/decoder ----
    parser.add_argument('--rae_model', default='RAE_DINOv2', type=str,
                        help='RAE encoder variant (currently only RAE_DINOv2)')
    parser.add_argument('--dinov2small', action='store_true', default=False,
                        help='Use DINOv2-Small (d=384); default is DINOv2-Base (d=768)')
    parser.add_argument('--reg_loss', action='store_true',
                        help='Enable joint [CLS]-patch modeling (paper §3.2)')
    parser.add_argument('--reg_loss_weight', type=float, default=0.2,
                        help='Weight lambda on the [CLS] auxiliary loss')
    parser.add_argument('--rae_normalize', action='store_true',
                        help='Element-wise standardize DINOv2 features (paper §3.1)')
    parser.add_argument('--normalization_stat_path', type=str, default=None,
                        help='Path to precomputed normalization statistics')

    # ---- training ----
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
    parser.add_argument('--batch_size', default=192, type=int,
                        help='Per-GPU batch size')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Absolute learning rate (overrides --blr)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base LR; actual LR = blr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        choices=['constant', 'cosine'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'])
    parser.add_argument('--muon_lr', type=float, default=0.001)
    parser.add_argument('--muon_momentum', type=float, default=0.95)
    parser.add_argument('--muon_weight_decay', type=float, default=0.1)
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='EMA decay used for sampling')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='Second EMA decay (tracked but unused for eval)')
    parser.add_argument('--P_mean', default=-0.8, type=float,
                        help='Logit-normal mean for the time schedule')
    parser.add_argument('--P_std', default=0.8, type=float,
                        help='Logit-normal std for the time schedule')
    parser.add_argument('--time_schedule', default='shift', type=str,
                        choices=['shift', 'logit_normal'],
                        help='shift = truncated logit-normal + dimension-aware time shift (paper default); '
                             'logit_normal = original JiT logit-normal')
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--sample_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float,
                        help='Probability of replacing the class label with the null class')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # ---- sampling ----
    parser.add_argument('--sampling_method', default='heun', type=str,
                        choices=['euler', 'heun'])
    parser.add_argument('--num_sampling_steps', default=25, type=int)
    parser.add_argument('--sample_schedule', default='time_shift', type=str,
                        choices=['uniform', 'edm', 'cosine', 'logsnr_uniform',
                                 'power2', 'time_shift', 'logit_normal'])
    parser.add_argument('--cfg', default=1.0, type=float)
    parser.add_argument('--cfg_cls', default=None, type=float,
                        help='CFG scale for [CLS] token; defaults to --cfg')
    parser.add_argument('--interval_min', default=0.0, type=float)
    parser.add_argument('--interval_max', default=1.0, type=float)
    parser.add_argument('--interval_cls_min', default=None, type=float)
    parser.add_argument('--interval_cls_max', default=None, type=float)
    parser.add_argument('--coupled_noise', action='store_true', default=False,
                        help='Inference-time: initialize [CLS] noise as spatial mean of patch noise')
    parser.add_argument('--coupled_noise_train', action='store_true', default=False,
                        help='Training-time variant of coupled_noise')
    parser.add_argument('--num_images', default=50000, type=int)
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Epoch interval between online evaluations')
    parser.add_argument('--online_eval', action='store_true', default=False)
    parser.add_argument('--evaluate_gen', action='store_true',
                        help='Skip training and only evaluate generation')
    parser.add_argument('--gen_bsz', type=int, default=256)
    parser.add_argument('--gen_precision', type=str, default='fp32', choices=['fp32', 'bf16'])
    parser.add_argument('--save_image_dir', type=str, default=None)
    parser.add_argument('--summary_file', type=str, default=None)

    # ---- dataset ----
    parser.add_argument('--data_path', default='./data/imagenet', type=str)
    parser.add_argument('--class_num', default=1000, type=int)

    # ---- checkpointing ----
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--resume', default='',
                        help='Directory containing checkpoint-last.pth to resume from')
    parser.add_argument('--checkpoint_path', default='',
                        help='Explicit checkpoint path (overrides --resume)')
    parser.add_argument('--save_last_freq', type=int, default=5)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda')

    # ---- distributed ----
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks   = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        print("Logging to:", args.output_dir)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "RiT"),
            config=vars(args),
            dir=args.output_dir,
        )
        wandb.define_metric("epoch_1000x")
        wandb.define_metric("train_*", step_metric="epoch_1000x")
        wandb.define_metric("lr", step_metric="epoch_1000x")
        wandb.define_metric("grad_norm", step_metric="epoch_1000x")
        wandb.define_metric("fid*", step_metric="epoch")
        wandb.define_metric("is*", step_metric="epoch")
        log_writer = True
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        log_writer = None

    if not args.evaluate_gen:
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
        )
        print("Sampler_train =", sampler_train)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True,
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    rae_kwargs = dict(
        rae_normalize=args.rae_normalize,
        normalization_stat_path=args.normalization_stat_path,
    )
    if args.rae_model == "RAE_DINOv2":
        rae_kwargs["dinov2small"] = args.dinov2small
    rae_model = RAE_models[args.rae_model](**rae_kwargs)

    args.rae_input_size = int(sqrt(rae_model.num_tokens))

    print("RAE =", rae_model)
    n_params = sum(p.numel() for p in rae_model.parameters() if p.requires_grad)
    print(f"Number of RAE parameters: {n_params / 1e6:.6f}M")

    rae_model.to(device)
    rae_model.requires_grad_(False)
    rae_model.eval()

    model = Denoiser(args, latent_dim=rae_model.latent_dim)
    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params / 1e6:.6f}M")

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual lr: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    if args.optimizer == 'muon':
        muon_params, adamw_params = split_params_for_muon(model_without_ddp)
        optimizer_adamw = torch.optim.AdamW(
            [{'params': adamw_params, 'weight_decay': 0.0}],
            lr=args.lr, betas=(0.9, 0.95),
        )
        optimizer_muon = torch.optim.Muon(
            muon_params, lr=args.muon_lr,
            momentum=args.muon_momentum, weight_decay=args.muon_weight_decay,
        )
        optimizer = [optimizer_adamw, optimizer_muon]
        print(optimizer_adamw)
        print(optimizer_muon)
    else:
        param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = (
        args.checkpoint_path if args.checkpoint_path
        else os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", checkpoint_path)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            if isinstance(optimizer, list):
                for opt, state in zip(optimizer, checkpoint['optimizer']):
                    opt.load_state_dict(state)
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    if args.evaluate_gen:
        print(f"Evaluating checkpoint at {args.start_epoch} epoch")
        print(f"CFG: {args.cfg}")
        if args.cfg != 1.0:
            print(f"Interval min: {args.interval_min} | Interval max: {args.interval_max}")
        print(f"Sampling method: {args.sampling_method} | Num images: {args.num_images} | Sampling steps: {args.num_sampling_steps}")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, rae_model, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, rae_model, data_loader_train,
                        optimizer, device, epoch, log_writer=log_writer, args=args)

        if epoch % args.save_last_freq == 0 or (epoch + 1 == args.epochs and epoch > 0):
            misc.save_model(args=args, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, epoch=epoch, epoch_name="last")

        if epoch % 20 == 0 and epoch > 0:
            misc.save_model(args=args, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, epoch=epoch)

        if args.online_eval and epoch > 0 and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, rae_model, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Output directory:", args.output_dir)
    main(args)
