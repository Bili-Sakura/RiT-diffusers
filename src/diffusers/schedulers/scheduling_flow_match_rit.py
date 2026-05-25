# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.schedulers.scheduling_utils import SchedulerMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "scheduler_config.json"

    class SchedulerMixin:
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


@dataclass
class RiTFlowMatchSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    prev_cls_sample: Optional[torch.FloatTensor] = None


def build_sample_timesteps(
    steps: int,
    schedule: str = "uniform",
    device=None,
    time_dist_shift: float = 1.0,
    p_mean: float = 0.0,
    p_std: float = 1.0,
) -> torch.Tensor:
    if schedule == "uniform":
        return torch.linspace(0.0, 1.0, steps + 1, device=device)

    if schedule == "edm":
        rho = 7.0
        sigma_min, sigma_max = 0.002, 80.0
        indices = torch.arange(steps + 1, device=device, dtype=torch.float64)
        sigma = (sigma_max ** (1.0 / rho) + (indices / steps) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
        timesteps = 1.0 / (1.0 + sigma)
        timesteps[0] = 0.0
        timesteps[-1] = 1.0
        return timesteps.float()

    if schedule == "cosine":
        indices = torch.arange(steps + 1, device=device, dtype=torch.float64)
        return (0.5 * (1 - torch.cos(math.pi * indices / steps))).float()

    if schedule == "logsnr_uniform":
        eps = 1e-3
        logit_lo = math.log(eps / (1 - eps))
        logit_hi = math.log((1 - eps) / eps)
        logits = torch.linspace(logit_lo, logit_hi, steps + 1, device=device)
        timesteps = torch.sigmoid(logits)
        timesteps[0] = 0.0
        timesteps[-1] = 1.0
        return timesteps

    if schedule == "power2":
        indices = torch.arange(steps + 1, device=device, dtype=torch.float64)
        return (indices / steps) ** 2

    if schedule == "logit_normal":
        u = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
        z = torch.erfinv(2 * u - 1) * math.sqrt(2)
        timesteps = torch.sigmoid(p_mean + p_std * z)
        timesteps[0] = 0.0
        timesteps[-1] = 1.0
        return timesteps.float()

    if schedule == "time_shift":
        u = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float64)
        u_shifted = u * time_dist_shift / (1.0 + (time_dist_shift - 1.0) * u)
        timesteps = 1.0 - u_shifted
        timesteps = timesteps.flip(0)
        timesteps[0] = 0.0
        timesteps[-1] = 1.0
        return timesteps.float()

    raise ValueError(f"Unknown sample schedule: {schedule}")


class RiTFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """Flow-matching scheduler for RiT with x-prediction and optional Heun sampling."""

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sample_schedule: str = "time_shift",
        sampling_method: str = "heun",
        pred_type: str = "x",
        sample_eps: float = 1e-5,
        noise_scale: float = 1.0,
        time_dist_shift: float = 0.0,
        latent_size: int = 16,
        latent_channels: int = 384,
        coupled_noise: bool = False,
    ):
        self.timesteps = None
        self._step_index = 0

    def _cfg(self, name: str):
        if hasattr(self.config, name):
            return getattr(self.config, name)
        return self.config[name]

    @property
    def time_dist_shift(self) -> float:
        shift = self._cfg("time_dist_shift")
        if shift and shift > 0:
            return shift
        latent_size = self._cfg("latent_size")
        latent_channels = self._cfg("latent_channels")
        return math.sqrt(latent_size * latent_size * latent_channels / 4096)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.timesteps = build_sample_timesteps(
            num_inference_steps,
            schedule=self._cfg("sample_schedule"),
            device=device,
            time_dist_shift=self.time_dist_shift,
        )
        self._step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def _to_velocity(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if self._cfg("pred_type") == "x":
            while timestep.ndim < sample.ndim:
                timestep = timestep.unsqueeze(-1)
            return (model_output - sample) / (1.0 - timestep).clamp_min(eps)
        return model_output

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        next_timestep: Union[float, torch.Tensor, None] = None,
        cls_model_output: Optional[torch.Tensor] = None,
        cls_sample: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[RiTFlowMatchSchedulerOutput, Tuple[torch.Tensor, ...]]:
        if next_timestep is None:
            raise ValueError("RiTFlowMatchScheduler.step requires `next_timestep`.")

        t = timestep
        t_next = next_timestep
        while t.ndim < sample.ndim:
            t = t.unsqueeze(-1)
        while t_next.ndim < sample.ndim:
            t_next = t_next.unsqueeze(-1)

        velocity = self._to_velocity(model_output, sample, t, self._cfg("sample_eps"))
        prev_sample = sample + (t_next - t) * velocity

        prev_cls_sample = None
        if cls_sample is not None and cls_model_output is not None:
            t_cls = t.squeeze(-1).squeeze(-1)
            t_next_cls = t_next.squeeze(-1).squeeze(-1)
            cls_velocity = self._to_velocity(cls_model_output, cls_sample, t_cls, self.config.sample_eps)
            prev_cls_sample = cls_sample + (t_next_cls - t_cls) * cls_velocity

        self._step_index += 1
        if not return_dict:
            return (prev_sample, prev_cls_sample)
        return RiTFlowMatchSchedulerOutput(prev_sample=prev_sample, prev_cls_sample=prev_cls_sample)

    def heun_step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        next_timestep: Union[float, torch.Tensor],
        model_output_fn,
        class_labels: torch.LongTensor,
        cls_model_output: Optional[torch.Tensor] = None,
        cls_sample: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[RiTFlowMatchSchedulerOutput, Tuple[torch.Tensor, ...]]:
        t = timestep
        t_next = next_timestep
        while t.ndim < sample.ndim:
            t = t.unsqueeze(-1)
        while t_next.ndim < sample.ndim:
            t_next = t_next.unsqueeze(-1)

        velocity_t = self._to_velocity(model_output, sample, t, self.config.sample_eps)
        euler_sample = sample + (t_next - t) * velocity_t

        if cls_sample is not None and cls_model_output is not None:
            t_cls = t.squeeze(-1).squeeze(-1)
            t_next_cls = t_next.squeeze(-1).squeeze(-1)
            cls_velocity_t = self._to_velocity(cls_model_output, cls_sample, t_cls, self.config.sample_eps)
            euler_cls = cls_sample + (t_next_cls - t_cls) * cls_velocity_t
            next_output, next_cls = model_output_fn(euler_sample, t_next, class_labels, cls_sample=euler_cls)
            velocity_next = self._to_velocity(next_output, euler_sample, t_next, self.config.sample_eps)
            cls_velocity_next = self._to_velocity(next_cls, euler_cls, t_next_cls, self.config.sample_eps)
            prev_sample = sample + (t_next - t) * 0.5 * (velocity_t + velocity_next)
            prev_cls_sample = cls_sample + (t_next_cls - t_cls) * 0.5 * (cls_velocity_t + cls_velocity_next)
        else:
            next_output, _ = model_output_fn(euler_sample, t_next, class_labels)
            velocity_next = self._to_velocity(next_output, euler_sample, t_next, self.config.sample_eps)
            prev_sample = sample + (t_next - t) * 0.5 * (velocity_t + velocity_next)
            prev_cls_sample = None

        if not return_dict:
            return (prev_sample, prev_cls_sample)
        return RiTFlowMatchSchedulerOutput(prev_sample=prev_sample, prev_cls_sample=prev_cls_sample)
