#!/usr/bin/env python3
"""Distributed latent mean / variance estimation for the frozen RAE encoder.

Used to produce the element-wise standardization statistics that RiT loads via
`--normalization_stat_path` (paper §3.1). Stats are accumulated over ImageNet
with a BatchNorm-like synchronized running-stat estimator, then saved as:

    {
      "mean": Tensor(C, H, W),
      "var":  Tensor(C, H, W),
      "mean_cls_token": Tensor(D,),
      "var_cls_token":  Tensor(D,),
    }
"""
import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rae import RAE_models


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, index


class LayerNormwStatistics(nn.Module):
    """BN-like running-statistics accumulator over batch dim only (for 3D/4D latents)."""

    def __init__(self, input_shape, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.input_shape = tuple(input_shape)

        self.gamma = nn.Parameter(torch.ones((1, *self.input_shape)))
        self.beta  = nn.Parameter(torch.zeros((1, *self.input_shape)))

        self.register_buffer("running_mean", torch.zeros(*self.input_shape))
        self.register_buffer("running_var",  torch.ones(*self.input_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1 + len(self.input_shape):
            raise ValueError(f"Expected x shape (B, {self.input_shape}), got {tuple(x.shape)}")
        if self.training:
            mean = x.mean(dim=0, keepdim=False)
            var  = x.var(dim=0, keepdim=False, unbiased=False)
            m = self.momentum
            self.running_mean.mul_(1 - m).add_(mean, alpha=m)
            self.running_var.mul_(1 - m).add_(var, alpha=m)
        else:
            mean = self.running_mean
            var  = self.running_var
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


def _extract_latents(rae, images: torch.Tensor):
    z, cls_token = rae.encode(images, return_cls_token=True)
    if cls_token.ndim == 2:
        cls_token = cls_token.unsqueeze(1)
    return z, cls_token


def _make_bn_for_latents(latents: torch.Tensor, eps: float = 1e-5, momentum: float = 0.1) -> nn.Module:
    nd = latents.ndim
    if nd == 2:
        return nn.BatchNorm1d(latents.shape[1], affine=False, track_running_stats=True, eps=eps, momentum=momentum)
    if nd in (3, 4):
        return LayerNormwStatistics(latents.shape[1:], eps=eps, momentum=momentum)
    if nd == 5:
        return nn.BatchNorm3d(latents.shape[1], affine=False, track_running_stats=True, eps=eps, momentum=momentum)
    raise ValueError(f"Unsupported latent shape {tuple(latents.shape)}")


@torch.no_grad()
def _sync_mean_var_across_ranks(mean_r: torch.Tensor, var_r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold per-rank running stats into a single global mean/var using
    mean = E[mean_r] and var = E[var_r] + Var(mean_r)."""
    if not dist.is_initialized():
        return mean_r, var_r
    world_size = dist.get_world_size()

    mean = mean_r.float()
    var  = var_r.float()
    mean_sum    = mean.clone()
    mean_sq_sum = mean * mean
    var_sum     = var.clone()

    dist.all_reduce(mean_sum,    op=dist.ReduceOp.SUM)
    dist.all_reduce(mean_sq_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(var_sum,     op=dist.ReduceOp.SUM)

    mean_global    = mean_sum / world_size
    mean_sq_global = mean_sq_sum / world_size
    var_of_mean    = mean_sq_global - mean_global * mean_global

    var_global = var_sum / world_size + var_of_mean
    var_global = torch.clamp(var_global, min=0.0)
    return mean_global, var_global


def _get_running_stats(module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    return module.running_mean.detach(), module.running_var.detach()


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script assumes CUDA + nccl DDP.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but this CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    rae_kwargs = dict(rae_normalize=False)
    if args.rae_model == "RAE_DINOv2":
        rae_kwargs["dinov2small"] = args.dinov2_small
    rae = RAE_models[args.rae_model](**rae_kwargs).to(device)
    rae.eval()

    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = IndexedImageFolder(args.data_path, transform=stage2_transform)
    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No images found at {args.data_path}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")
    base_ds = dataset if requested == total_available else Subset(dataset, list(range(requested)))

    sampler = DistributedSampler(base_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    loader = DataLoader(base_ds, batch_size=args.per_proc_batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=False, drop_last=False)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    # Auto-append encoder name to output path.
    model_suffix = args.rae_model
    if args.rae_model == "RAE_DINOv2":
        model_suffix = "RAE_DINOv2_small" if args.dinov2_small else "RAE_DINOv2_base"
    out_dir = os.path.join(args.sample_dir, model_suffix)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        print(f"[init] world_size={world_size} requested_samples={requested}")
        print(f"[path] saving stats under: {out_dir}")
    dist.barrier()

    first_batch = next(iter(loader), None)
    if first_batch is None:
        raise RuntimeError("Empty loader on this rank.")
    images0, _ = first_batch
    images0 = images0.to(device, non_blocking=True)
    with autocast(**autocast_kwargs):
        z0, cls_token0 = _extract_latents(rae, images0)

    stats_mod = _make_bn_for_latents(z0, eps=args.eps, momentum=args.momentum).to(device)
    stats_mod.train()

    stats_mod_cls_token = _make_bn_for_latents(cls_token0, eps=args.eps, momentum=args.momentum).to(device)
    stats_mod_cls_token.train()

    if rank == 0:
        print(f"[stats_mod] module={type(stats_mod).__name__} latent_shape={tuple(z0.shape)}")
    dist.barrier()

    for _ in range(1):
        iterator = tqdm(loader, desc="Latent stats", total=len(loader), disable=(rank != 0))
        for images, _indices in iterator:
            images = images.to(device, non_blocking=False)
            with autocast(**autocast_kwargs):
                z, cls_token = _extract_latents(rae, images)
            _ = stats_mod(z)
            _ = stats_mod_cls_token(cls_token)
            torch.cuda.synchronize()

    stats_mod.eval()
    stats_mod_cls_token.eval()
    mean_r, var_r = _get_running_stats(stats_mod)
    mean, var = _sync_mean_var_across_ranks(mean_r, var_r)
    mean_cls_token_r, var_cls_token_r = _get_running_stats(stats_mod_cls_token)
    mean_cls_token, var_cls_token = _sync_mean_var_across_ranks(mean_cls_token_r, var_cls_token_r)

    dist.barrier()
    if rank == 0:
        out_path = os.path.join(out_dir, "normalization_stats.pt")
        payload = {
            "mean": mean.cpu(),
            "var":  var.cpu(),
            "mean_cls_token": mean_cls_token.cpu(),
            "var_cls_token":  var_cls_token.cpu(),
        }
        torch.save(payload, out_path)
        print(f"[done] wrote: {out_path}")
        for k, v in payload.items():
            print(f"[{k}] shape={tuple(v.shape)} dtype={v.dtype}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to an ImageFolder directory (e.g. imagenet/train)")
    parser.add_argument("--sample-dir", type=str, default="stats/",
                        help="Base directory for stats output")
    parser.add_argument("--per-proc-batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of images to use (default: all)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.1)
    parser.add_argument("--dinov2-small", action="store_true", default=False,
                        help="Use DINOv2-Small instead of DINOv2-Base")
    parser.add_argument("--rae-model", type=str, default="RAE_DINOv2")
    args = parser.parse_args()
    main(args)
