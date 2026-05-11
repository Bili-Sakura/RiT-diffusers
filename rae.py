import torch
import torch.nn as nn
import torch.nn.functional as F
from rae_decoder import GeneralDecoder
from transformers import AutoConfig, AutoImageProcessor, Dinov2WithRegistersModel
from typing import Optional, Tuple
from math import sqrt


class Dinov2withNorm(nn.Module):
    """DINOv2-with-Registers encoder with optional post-LayerNorm affine disabled.

    When `normalize=True`, we strip the learned affine of the final LayerNorm so
    that every token satisfies ||z||^2 = d by construction (matching the paper's
    norm-concentration analysis).
    """

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = out.last_hidden_state[:, unused_token_num:]
        cls_token = out.last_hidden_state[:, 0]
        return image_features, cls_token


class RAE(nn.Module):
    """Representation AutoEncoder: frozen DINOv2 encoder + ViT-MAE style decoder.

    The encoder produces token features (B, N, D) which are optionally
    element-wise standardized to zero mean / unit variance using pre-computed
    stats. The decoder maps the (denormalized) features back to pixel space.
    """

    def __init__(
        self,
        # ---- encoder configs ----
        encoder_config_path: str = 'facebook/dinov2-with-registers-base',
        encoder_input_size: int = 224,
        normalize: bool = True,
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- reshape + normalization ----
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        rae_normalize: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = Dinov2withNorm(dinov2_path=encoder_config_path, normalize=normalize)
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, (
            f"encoder_input_size {self.encoder_input_size} must be divisible by "
            f"encoder_patch_size {self.encoder_patch_size}"
        )
        self.num_tokens = (self.encoder_input_size // self.encoder_patch_size) ** 2

        # Decoder
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.num_tokens))
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.num_tokens)
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if rae_normalize and normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.cls_token_mean = stats.get('mean_cls_token', None)
            self.cls_token_var = stats.get('var_cls_token', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            print("No normalization on RAE latents")
            self.do_normalization = False

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(self, x: torch.Tensor, return_cls_token: bool = False):
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size),
                                          mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z, cls_token = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        if return_cls_token:
            if self.do_normalization:
                cls_token_mean = self.cls_token_mean.to(cls_token.device) if self.cls_token_mean is not None else 0
                cls_token_var = self.cls_token_var.to(cls_token.device) if self.cls_token_var is not None else torch.ones_like(cls_token).to(cls_token.device)
                cls_token = (cls_token - cls_token_mean) / torch.sqrt(cls_token_var + self.eps)
            return z, cls_token
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = self.decoder(z).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def RAE_DINOv2(**kwargs) -> RAE:
    """DINOv2-with-Registers + ViT-MAE decoder.

    Set `dinov2small=True` for DINOv2-S (d=384, used in the main RiT-XL paper
    configuration); default is DINOv2-B (d=768).

    Pretrained decoder weights are loaded from `models/decoders/dinov2/...`; see
    the README for download instructions.
    """
    use_small = kwargs.pop("dinov2small", False)
    if use_small:
        print("Using DINOv2-Small as the RAE encoder")
        default_encoder_config_path = "facebook/dinov2-with-registers-small"
        default_decoder_config_path = "decoder_config/dinov2_small"
        default_pretrained_decoder_path = "models/decoders/dinov2/wReg_small/ViTXL_n08/model.pt"
        default_normalization_stat_path = "stats/RAE_DINOv2_small/normalization_stats.pt"
    else:
        print("Using DINOv2-Base as the RAE encoder")
        default_encoder_config_path = "facebook/dinov2-with-registers-base"
        default_decoder_config_path = "decoder_config/dinov2_base"
        default_pretrained_decoder_path = "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
        default_normalization_stat_path = "stats/RAE_DINOv2_base/normalization_stats.pt"
    normalization_stat_path = kwargs.pop("normalization_stat_path", default_normalization_stat_path)
    return RAE(
        encoder_config_path=default_encoder_config_path,
        encoder_input_size=224,
        normalize=True,
        decoder_config_path=default_decoder_config_path,
        pretrained_decoder_path=default_pretrained_decoder_path,
        noise_tau=0.0,
        reshape_to_2d=kwargs.pop("reshape_to_2d", True),
        normalization_stat_path=normalization_stat_path,
        **kwargs,
    )


RAE_models = {
    "RAE_DINOv2": RAE_DINOv2,
}
