#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from diffusers.models.autoencoders import AutoencoderRAE, create_rit_autoencoder
from diffusers.models.autoencoders.autoencoder_rae_presets import RIT_RAE_PRESETS, _load_latent_stats
from diffusers.models.transformers import RiTTransformer2DModel
from diffusers.schedulers import RiTFlowMatchScheduler


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "rit-xl": {
        "depth": 28,
        "hidden_size": 1152,
        "num_heads": 16,
        "in_context_len": 32,
        "in_context_start": 8,
        "patch_size": 1,
        "input_size": 16,
        "use_cls": True,
    },
}


def _load_state_dict(checkpoint_path: str, use_ema: bool = True) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        payload = safe_load_file(checkpoint_path, device="cpu")
        return _clean_state_dict(payload)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if use_ema and "model_ema1" in checkpoint:
            state_dict = checkpoint["model_ema1"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "transformer.", "net.")
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _save_config(output_dir: Path, config: Dict[str, Any], filename: str = "config.json"):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(state_dict, str(output_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path):
    model_index = {
        "_class_name": "RiTPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", "RiTFlowMatchScheduler"],
        "transformer": ["diffusers", "RiTTransformer2DModel"],
        "autoencoder": ["diffusers", "AutoencoderRAE"],
    }
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert original RiT checkpoints to a Diffusers pipeline directory.")
    parser.add_argument("--checkpoint", required=True, help="Path to RiT .pth/.pt/.safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="rit-xl")
    parser.add_argument("--in-channels", type=int, default=384)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--sample-schedule", default="time_shift")
    parser.add_argument("--sampling-method", choices=["euler", "heun"], default="heun")
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--dinov2-small", action="store_true", help="Use DINOv2-Small RAE preset.")
    parser.add_argument("--rae-normalize", action="store_true", help="Enable latent standardization in autoencoder.")
    parser.add_argument("--decoder-path", default=None, help="Optional pretrained ViT decoder weights.")
    parser.add_argument("--normalization-stat-path", default=None)
    parser.add_argument("--no-ema", action="store_true", help="Load raw model weights instead of model_ema1.")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"
    autoencoder_dir = output_dir / "autoencoder"

    state_dict = _load_state_dict(args.checkpoint, use_ema=not args.no_ema)
    transformer_config = {
        "_class_name": "RiTTransformer2DModel",
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        **MODEL_PRESETS[args.model_size],
    }

    if args.check_load:
        model = RiTTransformer2DModel(**{k: v for k, v in transformer_config.items() if k != "_class_name"})
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            raise SystemExit(1)

    _save_config(transformer_dir, transformer_config)
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    time_dist_shift = (16 * 16 * args.in_channels / 4096) ** 0.5
    _save_config(
        scheduler_dir,
        {
            "_class_name": "RiTFlowMatchScheduler",
            "num_train_timesteps": 1000,
            "sample_schedule": args.sample_schedule,
            "sampling_method": args.sampling_method,
            "pred_type": "x",
            "sample_eps": 1e-5,
            "noise_scale": 1.0,
            "time_dist_shift": time_dist_shift,
            "latent_size": 16,
            "latent_channels": args.in_channels,
            "coupled_noise": True,
        },
        filename="scheduler_config.json",
    )

    preset = "dinov2_small" if args.dinov2_small else "dinov2_base"
    stats_path = args.normalization_stat_path
    if stats_path is None and args.rae_normalize:
        stats_path = RIT_RAE_PRESETS[preset]["normalization_stat_path"]

    latents_mean, latents_std = _load_latent_stats(stats_path if args.rae_normalize else None)
    autoencoder = create_rit_autoencoder(
        preset=preset,
        rae_normalize=args.rae_normalize,
        pretrained_decoder_path=args.decoder_path,
        normalization_stat_path=stats_path,
    )
    autoencoder.save_pretrained(autoencoder_dir, safe_serialization=args.safe_serialization)

    if args.check_load:
        reloaded = AutoencoderRAE.from_pretrained(autoencoder_dir)
        if len(list(reloaded.parameters())) != len(list(autoencoder.parameters())):
            raise SystemExit("Autoencoder reload sanity check failed.")

    _write_model_index(output_dir)
    print(f"Saved Diffusers-style RiT pipeline to {output_dir}")


if __name__ == "__main__":
    main()
