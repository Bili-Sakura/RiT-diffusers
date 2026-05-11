"""Flow-matching denoiser wrapper for the RiT backbone.

Implements the training objective and ODE sampler from:
    "RiT: Vanilla Diffusion Transformers Are Enough in Representation Space."

Key design choices:
  - x-prediction: the network outputs the clean feature \\hat z_0 directly; the
    velocity is recovered as (\\hat z_0 - z_t) / (1 - t) and the loss is MSE on
    the velocity (equivalent to (1-t)^{-2}-weighted x-prediction loss).
  - Dimension-aware time shift: X ~ logit-normal, X' = X s / (1+(s-1)X), t=1-X',
    with s = sqrt(H W D / 4096). This pushes the median timestep towards the
    high-noise end to compensate for the per-token dimensionality.
  - Joint [CLS]-patch modeling: the [CLS] token is noised along the same
    flow-matching path, prepended to the patch sequence, and predicted by a
    separate linear head with its own loss weight (reg_loss_weight).
"""
import math

import torch
import torch.nn as nn

from model import RiT_models


def truncated_logitnormal_sample(shape, mu=0.0, sigma=1.0, device=None, time_dist_shift=2.0):
    """Sample t in [0,1] with Z = logit(X) ~ Normal(mu, sigma^2), then apply the
    dimension-aware time shift X' = X s / (1 + (s-1) X) and return t = 1 - X'.

    `time_dist_shift = s = sqrt(H W D / 4096)` (\\S3.3 of the paper).
    """
    low = 1e-3
    high = 1 - 1e-3
    mu    = torch.as_tensor(mu, device=device)
    sigma = torch.as_tensor(sigma, device=device)
    low   = torch.as_tensor(low, device=device)
    high  = torch.as_tensor(high, device=device)

    z_low  = torch.logit(low)
    z_high = torch.logit(high)

    base = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    alpha = (z_low  - mu) / sigma
    beta  = (z_high - mu) / sigma

    cdf_alpha = base.cdf(alpha)
    cdf_beta  = base.cdf(beta)

    out_shape = torch.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = torch.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)

    Z = mu + sigma * base.icdf(U)
    X = torch.sigmoid(Z)
    X = X * time_dist_shift / (1 + (time_dist_shift - 1) * X)
    return 1 - X.clamp(low, high)


def build_sample_timesteps(steps, schedule='uniform', device=None, time_dist_shift=1.0, P_mean=0.0, P_std=1.0):
    """Build timestep grid for ODE sampling from t=0 (noise) to t=1 (data).

    Supported schedules (see Appendix G of the paper for formulas):
        uniform, edm, cosine, logsnr_uniform, power2, logit_normal, time_shift
    """
    if schedule == 'uniform':
        return torch.linspace(0.0, 1.0, steps + 1, device=device)

    if schedule == 'edm':
        rho = 7.0
        sigma_min, sigma_max = 0.002, 80.0
        i = torch.arange(steps + 1, device=device, dtype=torch.float64)
        sigma = (sigma_max ** (1.0 / rho) + (i / steps) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
        t = 1.0 / (1.0 + sigma)
        t[0] = 0.0
        t[-1] = 1.0
        return t.float()

    if schedule == 'cosine':
        i = torch.arange(steps + 1, device=device, dtype=torch.float64)
        t = 0.5 * (1 - torch.cos(math.pi * i / steps))
        return t.float()

    if schedule == 'logsnr_uniform':
        eps = 1e-3
        logit_lo = math.log(eps / (1 - eps))
        logit_hi = math.log((1 - eps) / eps)
        logits = torch.linspace(logit_lo, logit_hi, steps + 1, device=device)
        t = torch.sigmoid(logits)
        t[0] = 0.0
        t[-1] = 1.0
        return t

    if schedule == 'power2':
        i = torch.arange(steps + 1, device=device, dtype=torch.float64)
        return (i / steps) ** 2

    if schedule == 'logit_normal':
        u = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float64)
        u = u.clamp(1e-6, 1 - 1e-6)
        z = torch.erfinv(2 * u - 1) * math.sqrt(2)
        t = torch.sigmoid(P_mean + P_std * z)
        t[0] = 0.0
        t[-1] = 1.0
        return t.float()

    if schedule == 'time_shift':
        s = time_dist_shift
        u = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float64)
        u_shifted = u * s / (1.0 + (s - 1.0) * u)
        t = 1.0 - u_shifted
        t = t.flip(0)
        t[0] = 0.0
        t[-1] = 1.0
        return t.float()

    raise ValueError(f"Unknown sample schedule: {schedule}")


class Denoiser(nn.Module):
    """Wraps a RiT backbone with the flow-matching training objective and an
    ODE sampler (Euler or Heun) with optional classifier-free guidance.
    """

    def __init__(self, args, latent_dim: int = 768):
        super().__init__()
        self.net = RiT_models[args.model](
            input_size=args.rae_input_size,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            reg_loss=args.reg_loss,
            in_channels=latent_dim,
        )
        self.img_size = args.rae_input_size
        self.latent_channels = latent_dim
        self.num_classes = args.class_num
        self.reg_loss = args.reg_loss
        self.reg_loss_weight = args.reg_loss_weight
        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.sample_eps = args.sample_eps
        self.noise_scale = args.noise_scale
        self.pred_type = getattr(args, 'pred_type', 'x')   # 'x' or 'v'
        self.time_schedule = getattr(args, 'time_schedule', 'shift')

        # EMA tracking (populated in main.py)
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # Generation hyper-parameters
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_cls_scale = args.cfg_cls if args.cfg_cls is not None else args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.cfg_cls_interval = (
            args.interval_cls_min if args.interval_cls_min is not None else args.interval_min,
            args.interval_cls_max if args.interval_cls_max is not None else args.interval_max,
        )
        self.coupled_noise = args.coupled_noise
        self.coupled_noise_train = args.coupled_noise_train
        self.sample_schedule = getattr(args, 'sample_schedule', 'uniform')

        # Dimension-aware time shift: s = sqrt(H W D / 4096).
        self.time_dist_shift = math.sqrt(self.img_size * self.img_size * self.latent_channels / 4096)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, n: int, device=None):
        """Original JiT-style logit-normal sampling (used when
        `time_schedule=logit_normal` for ablation)."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels, cls_token=None):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        if self.time_schedule == 'logit_normal':
            t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        else:
            t = truncated_logitnormal_sample(
                x.size(0), device=x.device, time_dist_shift=self.time_dist_shift
            ).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        if self.reg_loss and cls_token is not None:
            t_cls = t.squeeze(-1).squeeze(-1)
            if self.coupled_noise_train:
                e_cls = e.mean(dim=(-2, -1))
            else:
                e_cls = torch.randn_like(cls_token) * self.noise_scale
            z_cls = t_cls * cls_token + (1 - t_cls) * e_cls
            v_cls = (cls_token - z_cls) / (1 - t_cls).clamp_min(self.t_eps)
            net_out, net_cls_out = self.net(z, t.flatten(), labels_dropped, z_cls=z_cls)
        else:
            z_cls = None
            v_cls = None
            net_out = self.net(z, t.flatten(), labels_dropped)

        if self.pred_type == 'x':
            v_pred = (net_out - z) / (1 - t).clamp_min(self.t_eps)
        else:
            v_pred = net_out

        loss_fm = ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()
        loss_dict = {'loss_fm': loss_fm}

        if self.reg_loss and cls_token is not None:
            if self.pred_type == 'x':
                v_cls_pred = (net_cls_out - z_cls) / (1 - t_cls).clamp_min(self.t_eps)
            else:
                v_cls_pred = net_cls_out
            loss_cls = ((v_cls - v_cls_pred) ** 2).mean()
            loss_dict['loss_cls_raw'] = loss_cls

        loss = loss_fm
        if 'loss_cls_raw' in loss_dict:
            loss = loss + self.reg_loss_weight * loss_dict['loss_cls_raw']
        loss_dict['loss_total'] = loss
        return loss, loss_dict

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, self.latent_channels, self.img_size, self.img_size, device=device)
        if self.reg_loss:
            if self.coupled_noise:
                z_cls = z.mean(dim=(-2, -1))
            else:
                z_cls = self.noise_scale * torch.randn(bsz, self.latent_channels, device=device)
        else:
            z_cls = None

        timesteps = build_sample_timesteps(
            self.steps, self.sample_schedule, device=device,
            time_dist_shift=self.time_dist_shift,
        ).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == 'euler':
            stepper = self._euler_step
        elif self.method == 'heun':
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampler {self.method}")

        for i in range(self.steps - 1):
            z, z_cls = stepper(z, timesteps[i], timesteps[i + 1], labels, z_cls=z_cls)
        # Final Euler step (Heun would require an extra eval past t=1).
        z, _ = self._euler_step(z, timesteps[-2], timesteps[-1], labels, z_cls=z_cls)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, z_cls=None):
        # --- conditional branch ---
        if self.reg_loss and z_cls is not None:
            net_out_cond, net_cls_cond = self.net(z, t.flatten(), labels, z_cls=z_cls)
            if self.pred_type == 'x':
                v_cond = (net_out_cond - z) / (1.0 - t).clamp_min(self.sample_eps)
                v_cls_cond = (net_cls_cond - z_cls) / (1.0 - t.squeeze(-1).squeeze(-1)).clamp_min(self.sample_eps)
            else:
                v_cond = net_out_cond
                v_cls_cond = net_cls_cond
        else:
            net_out_cond = self.net(z, t.flatten(), labels)
            if self.pred_type == 'x':
                v_cond = (net_out_cond - z) / (1.0 - t).clamp_min(self.sample_eps)
            else:
                v_cond = net_out_cond
            v_cls_cond = None

        if self.cfg_scale == 1.0:
            return v_cond, v_cls_cond

        # --- unconditional branch ---
        if self.reg_loss and z_cls is not None:
            net_out_uncond, net_cls_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes), z_cls=z_cls)
            if self.pred_type == 'x':
                v_uncond = (net_out_uncond - z) / (1.0 - t).clamp_min(self.sample_eps)
                v_uncond_cls = (net_cls_uncond - z_cls) / (1.0 - t.squeeze(-1).squeeze(-1)).clamp_min(self.sample_eps)
            else:
                v_uncond = net_out_uncond
                v_uncond_cls = net_cls_uncond
        else:
            net_out_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
            if self.pred_type == 'x':
                v_uncond = (net_out_uncond - z) / (1.0 - t).clamp_min(self.sample_eps)
            else:
                v_uncond = net_out_uncond
            v_uncond_cls = None

        # Interval-limited CFG for patches.
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
        v_guided = v_uncond + cfg_scale_interval * (v_cond - v_uncond)

        # Interval-limited CFG for [CLS] (may differ from patch interval).
        if self.reg_loss and z_cls is not None:
            cls_low, cls_high = self.cfg_cls_interval
            cls_interval_mask = (t < cls_high) & ((cls_low == 0) | (t > cls_low))
            cfg_cls_interval = torch.where(cls_interval_mask, self.cfg_cls_scale, 1.0)
            v_cls_guided = v_uncond_cls + cfg_cls_interval.squeeze(-1).squeeze(-1) * (v_cls_cond - v_uncond_cls)
        else:
            v_cls_guided = None
        return v_guided, v_cls_guided

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, z_cls=None):
        if self.reg_loss and z_cls is not None:
            v_pred, v_cls_pred = self._forward_sample(z, t, labels, z_cls=z_cls)
            z_next = z + (t_next - t) * v_pred
            z_cls_next = z_cls + (t_next.squeeze(-1).squeeze(-1) - t.squeeze(-1).squeeze(-1)) * v_cls_pred
        else:
            v_pred, _ = self._forward_sample(z, t, labels)
            z_next = z + (t_next - t) * v_pred
            z_cls_next = None
        return z_next, z_cls_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, z_cls=None):
        if self.reg_loss and z_cls is not None:
            v_pred_t, v_cls_pred_t = self._forward_sample(z, t, labels, z_cls=z_cls)
            z_next_euler = z + (t_next - t) * v_pred_t
            z_cls_next_euler = z_cls + (t_next.squeeze(-1).squeeze(-1) - t.squeeze(-1).squeeze(-1)) * v_cls_pred_t
            v_pred_t_next, v_cls_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, z_cls=z_cls_next_euler)
            v_pred = 0.5 * (v_pred_t + v_pred_t_next)
            v_cls_pred = 0.5 * (v_cls_pred_t + v_cls_pred_t_next)
            z_next = z + (t_next - t) * v_pred
            z_cls_next = z_cls + (t_next.squeeze(-1).squeeze(-1) - t.squeeze(-1).squeeze(-1)) * v_cls_pred
        else:
            v_pred_t, _ = self._forward_sample(z, t, labels)
            z_next_euler = z + (t_next - t) * v_pred_t
            v_pred_t_next, _ = self._forward_sample(z_next_euler, t_next, labels)
            v_pred = 0.5 * (v_pred_t + v_pred_t_next)
            z_next = z + (t_next - t) * v_pred
            z_cls_next = None
        return z_next, z_cls_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
