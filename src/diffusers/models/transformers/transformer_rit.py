# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class _Config(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

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
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            init(self, *args, **kwargs)

        return wrapper


from .rit_utils import (
    RiTRMSNorm,
    RiTVisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)


def _modulate(hidden_states, shift, scale):
    return hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass
class RiTTransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor
    cls_sample: Optional[torch.FloatTensor] = None


class RiTPatchEmbed(nn.Module):
    def __init__(self, input_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return hidden_states.flatten(2).transpose(1, 2)


class RiTTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps):
        return self.mlp(self.timestep_embedding(timesteps, self.frequency_embedding_size))


class RiTLabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, class_labels):
        return self.embedding_table(class_labels)


class RiTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RiTRMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RiTRMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, hidden_states, rope):
        batch_size, seq_len, channels = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)
        query = rope(query)
        key = rope(key)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, channels)
        hidden_states = self.proj(hidden_states)
        return self.proj_drop(hidden_states)


class RiTSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, hidden_states):
        x12 = self.w12(hidden_states)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class RiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_cls=False):
        super().__init__()
        self.use_cls = use_cls
        self.norm_final = RiTRMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if use_cls:
            self.linear_cls = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, hidden_states, conditioning, cls=None):
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        hidden_states = _modulate(self.norm_final(hidden_states), shift, scale)
        if not self.use_cls and cls is None:
            return self.linear(hidden_states), None
        cls_token = self.linear_cls(hidden_states[:, 0]).unsqueeze(1)
        patch_states = self.linear(hidden_states[:, 1:])
        return patch_states, cls_token.squeeze(1)


class RiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RiTRMSNorm(hidden_size, eps=1e-6)
        self.attn = RiTAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RiTRMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = RiTSwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, hidden_states, conditioning, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(
            6, dim=-1
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(hidden_states), shift_msa, scale_msa), rope=feat_rope
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        )
        return hidden_states


class RiTTransformer2DModel(ModelMixin, ConfigMixin):
    """Vanilla Diffusion Transformer over DINOv2 representation features."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 384,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
        in_context_len: int = 32,
        in_context_start: int = 8,
        use_cls: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.use_cls = use_cls

        self.t_embedder = RiTTimestepEmbedder(hidden_size)
        self.y_embedder = RiTLabelEmbedder(num_classes, hidden_size)
        self.x_embedder = RiTPatchEmbed(input_size, patch_size, in_channels, hidden_size)

        num_patches = self.x_embedder.num_patches
        if use_cls:
            self.cls_projectors = nn.Linear(in_channels, hidden_size, bias=True)
            num_patches += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=0.02)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = RiTVisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=(1 if use_cls else 0),
        )
        self.feat_rope_incontext = RiTVisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len + (1 if use_cls else 0),
        )

        self.blocks = nn.ModuleList(
            [
                RiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = RiTFinalLayer(hidden_size, patch_size, self.out_channels, use_cls=use_cls)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
            cls_token=1 if self.use_cls else 0,
            extra_tokens=1 if self.use_cls else 0,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, hidden_states, patch_size):
        channels = self.out_channels
        height = width = int(hidden_states.shape[1] ** 0.5)
        assert height * width == hidden_states.shape[1]
        hidden_states = hidden_states.reshape(hidden_states.shape[0], height, width, patch_size, patch_size, channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(hidden_states.shape[0], channels, height * patch_size, height * patch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: torch.LongTensor,
        cls_token: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[RiTTransformer2DModelOutput, Tuple[torch.Tensor, ...]]:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=hidden_states.dtype)
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype).flatten()
        if timestep.numel() == 1:
            timestep = timestep.repeat(hidden_states.shape[0])

        class_labels = class_labels.to(device=hidden_states.device, dtype=torch.long).flatten()

        conditioning = self.t_embedder(timestep) + self.y_embedder(class_labels)
        tokens = self.x_embedder(hidden_states)
        if self.use_cls and cls_token is not None:
            cls_token = self.cls_projectors(cls_token).unsqueeze(1)
            tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.pos_embed

        for index, block in enumerate(self.blocks):
            if self.in_context_len > 0 and index == self.in_context_start:
                in_context_tokens = conditioning.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                tokens = torch.cat([in_context_tokens, tokens], dim=1)
            tokens = block(
                tokens,
                conditioning,
                self.feat_rope if index < self.in_context_start else self.feat_rope_incontext,
            )
        tokens = tokens[:, self.in_context_len :]
        patch_tokens, cls_pred = self.final_layer(tokens, conditioning, cls=cls_token)
        sample = self.unpatchify(patch_tokens, self.patch_size)

        if not return_dict:
            if self.use_cls and cls_token is not None:
                return (sample, cls_pred)
            return (sample,)
        return RiTTransformer2DModelOutput(
            sample=sample,
            cls_sample=cls_pred if self.use_cls and cls_token is not None else None,
        )
