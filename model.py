# --------------------------------------------------------
# References:
# JiT:           https://github.com/LTH14/JiT
# RAE:           https://github.com/bytetriper/RAE
# SiT:           https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
"""RiT model: a vanilla Diffusion Transformer operating on DINOv2 features.

Implements the architecture of:
    "RiT: Vanilla Diffusion Transformers Are Enough in Representation Space."

Key components:
  - vanilla PatchEmbed (patch_size=1 over 16x16 DINOv2 grid)
  - SwiGLU FFN, RMSNorm, QK-normalized attention, 2D VisionRoPE
  - adaLN modulation for timestep + class conditioning
  - 32 in-context class tokens injected at an intermediate layer (from JiT)
  - optional [CLS] token projection + separate linear head for joint [CLS]-patch
    modeling (enabled by `reg_loss=True`)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations via sinusoidal + MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Class-label embedding. The (num_classes)-th entry is the null/unconditional
    class used for classifier-free guidance."""

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)


class Attention(nn.Module):
    """Multi-head self-attention with QK RMSNorm and 2D rotary position embedding."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU feedforward block: FFN(x) = (SiLU(x W1) ⊙ x W3) W2."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """Final adaLN-modulated RMSNorm + linear head. When `reg_loss=True`, a
    separate linear head predicts the [CLS] token from its own position in the
    sequence."""

    def __init__(self, hidden_size, patch_size, out_channels, reg_loss=False):
        super().__init__()
        self.reg_loss = reg_loss
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if reg_loss:
            self.linear_cls = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c, cls=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        if not self.reg_loss and cls is None:
            return self.linear(x), None
        cls_token = self.linear_cls(x[:, 0]).unsqueeze(1)
        x = self.linear(x[:, 1:])
        return x, cls_token.squeeze(1)


class RiTBlock(nn.Module):
    """A single RiT transformer block: adaLN modulation + MHSA + SwiGLU FFN."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class RiT(nn.Module):
    """Representation Image Transformer — a vanilla DiT over DINOv2 features.

    Args:
        input_size: spatial resolution of the feature grid (typically 16).
        patch_size: DiT-side patch size. For DINOv2 features this is 1 (tokens
            are already patch-shaped).
        in_channels: feature dimension of the DINOv2 encoder (384 for Small,
            768 for Base).
        hidden_size, depth, num_heads, mlp_ratio: DiT transformer config.
        in_context_len, in_context_start: 32 learnable class-conditioned tokens
            injected at `in_context_start` and processed by all subsequent
            layers; discarded before the final projection (from JiT).
        reg_loss: if True, project the encoder's [CLS] token into the sequence
            and predict it from its own output position via `linear_cls`. This
            enables the joint [CLS]-patch modeling of RiT §3.2.
    """

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 768,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
        in_context_len: int = 32,
        in_context_start: int = 8,
        reg_loss: bool = False,
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
        self.reg_loss = reg_loss

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        self.x_embedder = PatchEmbed(img_size=input_size, patch_size=patch_size,
                                     in_chans=in_channels, embed_dim=hidden_size)

        num_patches = self.x_embedder.num_patches
        if reg_loss:
            # One extra position for the projected [CLS] token prepended to the sequence.
            self.cls_projectors = nn.Linear(in_channels, hidden_size, bias=True)
            num_patches += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len,
            num_cls_token=(1 if reg_loss else 0),
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len + (1 if reg_loss else 0),
        )

        # Attention & projection dropout is only active in the middle 50% of
        # layers; the default config uses drop=0 so this is a no-op.
        self.blocks = nn.ModuleList([
            RiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, reg_loss=reg_loss)

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
            int(self.x_embedder.num_patches ** 0.5),
            cls_token=1 if self.reg_loss else 0,
            extra_tokens=1 if self.reg_loss else 0,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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

    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y, z_cls=None):
        """
        x:     (N, C, H, W) noisy latent
        t:     (N,)         flow-matching timestep in [0, 1]
        y:     (N,)         class labels
        z_cls: (N, D) or None — noisy [CLS] token for joint modeling
        """
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        if self.reg_loss and z_cls is not None:
            z_cls = self.cls_projectors(z_cls).unsqueeze(1)
            x = torch.cat([z_cls, x], dim=1)
        x = x + self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)
        x = x[:, self.in_context_len:]

        x, cls_pred = self.final_layer(x, c, cls=z_cls)
        output = self.unpatchify(x, self.patch_size)

        if self.reg_loss and z_cls is not None:
            return output, cls_pred
        return output


# ----------------------------- Model registry --------------------------------- #

def RiT_S(**kwargs):
    in_channels = kwargs.pop('in_channels', 384)
    return RiT(depth=12, hidden_size=384, num_heads=6,
               in_context_len=32, in_context_start=4,
               patch_size=1, in_channels=in_channels, **kwargs)


def RiT_B(**kwargs):
    in_channels = kwargs.pop('in_channels', 768)
    return RiT(depth=12, hidden_size=768, num_heads=12,
               in_context_len=32, in_context_start=4,
               patch_size=1, in_channels=in_channels, **kwargs)


def RiT_L(**kwargs):
    in_channels = kwargs.pop('in_channels', 768)
    return RiT(depth=24, hidden_size=1024, num_heads=16,
               in_context_len=32, in_context_start=8,
               patch_size=1, in_channels=in_channels, **kwargs)


def RiT_XL(**kwargs):
    in_channels = kwargs.pop('in_channels', 768)
    return RiT(depth=28, hidden_size=1152, num_heads=16,
               in_context_len=32, in_context_start=8,
               patch_size=1, in_channels=in_channels, **kwargs)


def RiT_H(**kwargs):
    in_channels = kwargs.pop('in_channels', 768)
    return RiT(depth=32, hidden_size=1280, num_heads=16,
               in_context_len=32, in_context_start=10,
               patch_size=1, in_channels=in_channels, **kwargs)


RiT_models = {
    'RiT-S/16':  RiT_S,
    'RiT-B/16':  RiT_B,
    'RiT-L/16':  RiT_L,
    'RiT-XL/16': RiT_XL,
    'RiT-H/16':  RiT_H,
}
