# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RiT-specific helpers around upstream `AutoencoderRAE`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoImageProcessor, Dinov2WithRegistersModel

from .autoencoder_rae import AutoencoderRAE


RIT_RAE_PRESETS: Dict[str, Dict[str, Any]] = {
    "dinov2_small": {
        "encoder_pretrained_model_name_or_path": "facebook/dinov2-with-registers-small",
        "encoder_hidden_size": 384,
        "encoder_patch_size": 14,
        "encoder_num_hidden_layers": 12,
        "decoder_config_path": "decoder_config/dinov2_small",
        "pretrained_decoder_path": "models/decoders/dinov2/wReg_small/ViTXL_n08/model.pt",
        "normalization_stat_path": "stats/RAE_DINOv2_small/normalization_stats.pt",
    },
    "dinov2_base": {
        "encoder_pretrained_model_name_or_path": "facebook/dinov2-with-registers-base",
        "encoder_hidden_size": 768,
        "encoder_patch_size": 14,
        "encoder_num_hidden_layers": 12,
        "decoder_config_path": "decoder_config/dinov2_base",
        "pretrained_decoder_path": "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        "normalization_stat_path": "stats/RAE_DINOv2_base/normalization_stats.pt",
    },
}


def _load_decoder_hyperparameters(decoder_config_path: str) -> Dict[str, Any]:
    config_path = Path(decoder_config_path) / "config.json"
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_latent_stats(
    normalization_stat_path: Optional[str],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if normalization_stat_path is None:
        return None, None
    stats = torch.load(normalization_stat_path, map_location="cpu", weights_only=True)
    mean = stats.get("mean")
    var = stats.get("var")
    if mean is None or var is None:
        return None, None
    return mean, torch.sqrt(var + 1e-5)


def _load_pretrained_encoder(
    model: AutoencoderRAE,
    encoder_pretrained_model_name_or_path: str,
) -> None:
    encoder = Dinov2WithRegistersModel.from_pretrained(encoder_pretrained_model_name_or_path)
    encoder.requires_grad_(False)
    encoder.layernorm.elementwise_affine = False
    encoder.layernorm.weight = None
    encoder.layernorm.bias = None
    model.encoder.load_state_dict(encoder.state_dict(), strict=False)


def create_rit_autoencoder(
    preset: str = "dinov2_small",
    *,
    dinov2_small: Optional[bool] = None,
    rae_normalize: bool = False,
    pretrained_decoder_path: Optional[str] = None,
    normalization_stat_path: Optional[str] = None,
    encoder_pretrained_model_name_or_path: Optional[str] = None,
    decoder_config_path: Optional[str] = None,
    encoder_input_size: int = 224,
    reshape_to_2d: bool = True,
    noise_tau: float = 0.0,
    **kwargs,
) -> AutoencoderRAE:
    """Build an `AutoencoderRAE` with RiT paper presets and optional weight loading."""
    if dinov2_small is not None:
        preset = "dinov2_small" if dinov2_small else "dinov2_base"
    if preset not in RIT_RAE_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(RIT_RAE_PRESETS)}")

    preset_cfg = dict(RIT_RAE_PRESETS[preset])
    encoder_hub_path = encoder_pretrained_model_name_or_path or preset_cfg["encoder_pretrained_model_name_or_path"]
    decoder_config_path = decoder_config_path or preset_cfg["decoder_config_path"]
    decoder_path = pretrained_decoder_path or preset_cfg["pretrained_decoder_path"]
    stats_path = normalization_stat_path if rae_normalize else None
    if stats_path is None and rae_normalize:
        stats_path = preset_cfg["normalization_stat_path"]

    processor = AutoImageProcessor.from_pretrained(encoder_hub_path)
    decoder_cfg = _load_decoder_hyperparameters(decoder_config_path)
    latents_mean, latents_std = _load_latent_stats(stats_path)

    model = AutoencoderRAE(
        encoder_type="dinov2",
        encoder_hidden_size=preset_cfg["encoder_hidden_size"],
        encoder_patch_size=preset_cfg["encoder_patch_size"],
        encoder_num_hidden_layers=preset_cfg["encoder_num_hidden_layers"],
        decoder_hidden_size=decoder_cfg["decoder_hidden_size"],
        decoder_num_hidden_layers=decoder_cfg["decoder_num_hidden_layers"],
        decoder_num_attention_heads=decoder_cfg["decoder_num_attention_heads"],
        decoder_intermediate_size=decoder_cfg["decoder_intermediate_size"],
        patch_size=16,
        encoder_input_size=encoder_input_size,
        num_channels=3,
        encoder_norm_mean=processor.image_mean,
        encoder_norm_std=processor.image_std,
        latents_mean=latents_mean,
        latents_std=latents_std,
        noise_tau=noise_tau,
        reshape_to_2d=reshape_to_2d,
        **kwargs,
    )
    _load_pretrained_encoder(model, encoder_hub_path)
    if decoder_path is not None and Path(decoder_path).exists():
        decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=True)
        model.decoder.load_state_dict(decoder_state, strict=False)
    return model
