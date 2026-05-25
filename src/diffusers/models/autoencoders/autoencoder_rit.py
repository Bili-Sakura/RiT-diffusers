# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor, Dinov2WithRegistersModel

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(nn.Module):
        pass

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = {key: value for key, value in bound.arguments.items() if key != "self"}
            init(self, *args, **kwargs)

        return wrapper


from .vit_mae_decoder import GeneralDecoder


@dataclass
class RiTAutoencoderOutput(BaseOutput):
    sample: torch.FloatTensor


class Dinov2WithNorm(nn.Module):
    def __init__(self, dinov2_path: str, normalize: bool = True):
        super().__init__()
        self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        unused_token_num = 5
        image_features = outputs.last_hidden_state[:, unused_token_num:]
        cls_token = outputs.last_hidden_state[:, 0]
        return image_features, cls_token


class RiTAutoencoderKL(ModelMixin, ConfigMixin):
    """Representation autoencoder: frozen DINOv2 encoder and ViT-MAE decoder."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        encoder_pretrained_model_name_or_path: str = "facebook/dinov2-with-registers-small",
        encoder_input_size: int = 224,
        normalize_encoder: bool = True,
        decoder_config_path: str = "decoder_config/dinov2_small",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        rae_normalize: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = Dinov2WithNorm(dinov2_path=encoder_pretrained_model_name_or_path, normalize=normalize_encoder)
        processor = AutoImageProcessor.from_pretrained(encoder_pretrained_model_name_or_path)
        self.register_buffer("encoder_mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("encoder_std", torch.tensor(processor.image_std).view(1, 3, 1, 1), persistent=False)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        if self.encoder_input_size % self.encoder_patch_size != 0:
            raise ValueError("encoder_input_size must be divisible by encoder_patch_size")
        self.num_tokens = (self.encoder_input_size // self.encoder_patch_size) ** 2

        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.num_tokens))
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.num_tokens)
        if pretrained_decoder_path is not None:
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu", weights_only=True)
            self.decoder.load_state_dict(state_dict, strict=False)

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if rae_normalize and normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu", weights_only=True)
            self.register_buffer("latent_mean", stats.get("mean"), persistent=False)
            self.register_buffer("latent_var", stats.get("var"), persistent=False)
            self.register_buffer("cls_token_mean", stats.get("mean_cls_token"), persistent=False)
            self.register_buffer("cls_token_var", stats.get("var_cls_token"), persistent=False)
            self.do_normalization = True
            self.eps = eps
        else:
            self.do_normalization = False
            self.eps = eps

    def _noising(self, hidden_states: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (hidden_states.size(0),) + (1,) * (len(hidden_states.shape) - 1), device=hidden_states.device
        )
        return hidden_states + noise_sigma * torch.randn_like(hidden_states)

    def encode(
        self,
        pixel_values: torch.Tensor,
        return_cls_token: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _, _, height, width = pixel_values.shape
        if height != self.encoder_input_size or width != self.encoder_input_size:
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        pixel_values = (pixel_values - self.encoder_mean) / self.encoder_std
        latents, cls_token = self.encoder(pixel_values)
        if self.training and self.noise_tau > 0:
            latents = self._noising(latents)
        if self.reshape_to_2d:
            batch_size, num_tokens, channels = latents.shape
            side = int(sqrt(num_tokens))
            latents = latents.transpose(1, 2).view(batch_size, channels, side, side)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(latents.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(latents.device) if self.latent_var is not None else 1
            latents = (latents - latent_mean) / torch.sqrt(latent_var + self.eps)
        if return_cls_token:
            if self.do_normalization:
                cls_mean = self.cls_token_mean.to(cls_token.device) if self.cls_token_mean is not None else 0
                cls_var = (
                    self.cls_token_var.to(cls_token.device)
                    if self.cls_token_var is not None
                    else torch.ones_like(cls_token).to(cls_token.device)
                )
                cls_token = (cls_token - cls_mean) / torch.sqrt(cls_var + self.eps)
            return latents, cls_token
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(latents.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(latents.device) if self.latent_var is not None else 1
            latents = latents * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            batch_size, channels, height, width = latents.shape
            num_tokens = height * width
            latents = latents.view(batch_size, channels, num_tokens).transpose(1, 2)
        logits = self.decoder(latents).logits
        reconstruction = self.decoder.unpatchify(logits)
        return reconstruction * self.encoder_std + self.encoder_mean

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[RiTAutoencoderOutput, torch.Tensor]:
        latents = self.encode(pixel_values)
        sample = self.decode(latents)
        if not return_dict:
            return (sample,)
        return RiTAutoencoderOutput(sample=sample)
