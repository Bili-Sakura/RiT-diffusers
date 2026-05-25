# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

try:
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.utils import BaseOutput
    from diffusers.utils.torch_utils import randn_tensor
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class DiffusionPipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._execution_device = torch.device("cpu")

        def register_modules(self, **kwargs):
            for name, module in kwargs.items():
                setattr(self, name, module)

        def to(self, device):
            self._execution_device = torch.device(device)
            return super().to(device)

        def progress_bar(self, iterable):
            return iterable

        def maybe_free_model_hooks(self):
            return None

    class VaeImageProcessor:
        def postprocess(self, image, output_type="pil"):
            if output_type == "pt":
                return image
            if output_type == "np":
                return image.detach().cpu().numpy()
            images = []
            for sample in image:
                sample = sample.detach().float().cpu().clamp(0, 1)
                sample = sample.permute(1, 2, 0).numpy()
                images.append(Image.fromarray((sample * 255).round().astype("uint8")))
            return images

    def randn_tensor(shape, generator=None, device=None, dtype=None):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)


from ...models.autoencoders.autoencoder_rae import AutoencoderRAE
from ...models.autoencoders.autoencoder_rae import create_rit_autoencoder
from ...models.transformers.transformer_rit import RiTTransformer2DModel
from ...schedulers.scheduling_flow_match_rit import RiTFlowMatchScheduler


class RiTPipelineOutput(BaseOutput):
    images: List[Image.Image]
    latents: Optional[torch.Tensor] = None


class RiTPipeline(DiffusionPipeline):
    """Class-conditional RiT pipeline with RAE decode."""

    model_cpu_offload_seq = "transformer->autoencoder"
    _optional_components = []

    def __init__(
        self,
        transformer: RiTTransformer2DModel,
        scheduler: RiTFlowMatchScheduler,
        autoencoder: AutoencoderRAE,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, autoencoder=autoencoder)
        self.image_processor = VaeImageProcessor()
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        autoencoder_subfolder = model_kwargs.pop("autoencoder_subfolder", None)
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"
        if autoencoder_subfolder is None and (base_path / "autoencoder").exists():
            autoencoder_subfolder = "autoencoder"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            transformer_path = str(base_path / transformer_subfolder) if transformer_subfolder else pretrained_model_name_or_path
            transformer = RiTTransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
            try:
                scheduler = RiTFlowMatchScheduler.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=scheduler_subfolder,
                )
            except Exception:
                scheduler = RiTFlowMatchScheduler()
            autoencoder_path = str(base_path / autoencoder_subfolder) if autoencoder_subfolder else None
            if autoencoder_path is not None:
                try:
                    autoencoder = AutoencoderRAE.from_pretrained(autoencoder_path, **model_kwargs)
                except Exception:
                    autoencoder = create_rit_autoencoder(dinov2_small=True)
            else:
                autoencoder = create_rit_autoencoder(dinov2_small=True)
            id2label = cls._read_id2label_from_model_index(str(base_path))
            return cls(transformer=transformer, scheduler=scheduler, autoencoder=autoencoder, id2label=id2label)

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
        if not variant_path:
            return {}
        model_index_path = Path(variant_path).resolve() / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def id2label(self) -> Dict[int, str]:
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        if isinstance(label, str):
            label = [label]
        missing = [item for item in label if item not in self.labels]
        if missing:
            raise ValueError(f"Unknown label(s): {missing}")
        return [self.labels[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
    ) -> torch.LongTensor:
        if torch.is_tensor(class_labels):
            return class_labels.to(device=self._execution_device, dtype=torch.long).reshape(-1)
        if isinstance(class_labels, int):
            class_label_ids = [class_labels]
        elif isinstance(class_labels, str):
            class_label_ids = self.get_label_ids(class_labels)
        elif class_labels and isinstance(class_labels[0], str):
            class_label_ids = self.get_label_ids(class_labels)
        else:
            class_label_ids = list(class_labels)
        return torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)

    def prepare_latents(
        self,
        batch_size: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        latent_size = self.transformer.config.input_size
        latent_channels = self.transformer.config.in_channels
        latents = randn_tensor(
            (batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noise_scale = self.scheduler._cfg("noise_scale")
        latents = latents * noise_scale
        cls_latents = None
        if self.transformer.config.use_cls:
            if self.scheduler._cfg("coupled_noise"):
                cls_latents = latents.mean(dim=(-2, -1))
            else:
                cls_latents = randn_tensor(
                    (batch_size, latent_channels),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                ) * noise_scale
        return latents, cls_latents

    def _apply_cfg(
        self,
        cond_output: torch.Tensor,
        uncond_output: torch.Tensor,
        guidance_scale: float,
        timestep: torch.Tensor,
        guidance_interval: Tuple[float, float],
    ) -> torch.Tensor:
        if guidance_scale <= 1.0:
            return cond_output
        low, high = guidance_interval
        interval_mask = (timestep < high) & ((low == 0) | (timestep > low))
        while interval_mask.ndim < cond_output.ndim:
            interval_mask = interval_mask.unsqueeze(-1)
        cfg_scale = torch.where(interval_mask, guidance_scale, 1.0)
        return uncond_output + cfg_scale * (cond_output - uncond_output)

    def _forward_model(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        cls_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = self.transformer(
            latents,
            timestep,
            class_labels,
            cls_token=cls_latents,
            return_dict=True,
        )
        return output.sample, output.cls_sample

    def _predict_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        guidance_scale: float,
        guidance_interval: Tuple[float, float],
        guidance_cls_scale: float,
        guidance_cls_interval: Tuple[float, float],
        cls_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cond_output, cond_cls = self._forward_model(latents, timestep, class_labels, cls_latents=cls_latents)
        if guidance_scale <= 1.0 and guidance_cls_scale <= 1.0:
            return cond_output, cond_cls

        null_labels = torch.full_like(class_labels, self.transformer.config.num_classes)
        uncond_output, uncond_cls = self._forward_model(latents, timestep, null_labels, cls_latents=cls_latents)

        guided = self._apply_cfg(cond_output, uncond_output, guidance_scale, timestep, guidance_interval)
        guided_cls = None
        if cond_cls is not None and uncond_cls is not None:
            guided_cls = self._apply_cfg(cond_cls, uncond_cls, guidance_cls_scale, timestep, guidance_cls_interval)
        return guided, guided_cls

    def _euler_update(
        self,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        cls_latents: Optional[torch.Tensor] = None,
        cls_model_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        step_output = self.scheduler.step(
            model_output,
            timestep,
            latents,
            next_timestep=next_timestep,
            cls_model_output=cls_model_output,
            cls_sample=cls_latents,
            return_dict=True,
        )
        return step_output.prev_sample, step_output.prev_cls_sample

    def _heun_update(
        self,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        guidance_scale: float,
        guidance_interval: Tuple[float, float],
        guidance_cls_scale: float,
        guidance_cls_interval: Tuple[float, float],
        cls_latents: Optional[torch.Tensor] = None,
        cls_model_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        euler_latents, euler_cls = self._euler_update(
            latents,
            model_output,
            timestep,
            next_timestep,
            cls_latents=cls_latents,
            cls_model_output=cls_model_output,
        )
        next_output, next_cls = self._predict_with_cfg(
            euler_latents,
            next_timestep,
            class_labels,
            guidance_scale,
            guidance_interval,
            guidance_cls_scale,
            guidance_cls_interval,
            cls_latents=euler_cls,
        )
        sample_eps = self.scheduler._cfg("sample_eps")
        velocity_t = self.scheduler._to_velocity(model_output, latents, timestep, sample_eps)
        velocity_next = self.scheduler._to_velocity(next_output, euler_latents, next_timestep, sample_eps)
        while timestep.ndim < latents.ndim:
            timestep = timestep.unsqueeze(-1)
        while next_timestep.ndim < latents.ndim:
            next_timestep = next_timestep.unsqueeze(-1)
        prev_latents = latents + (next_timestep - timestep) * 0.5 * (velocity_t + velocity_next)

        prev_cls = None
        if cls_latents is not None and cls_model_output is not None and next_cls is not None:
            t_cls = timestep.squeeze(-1).squeeze(-1)
            t_next_cls = next_timestep.squeeze(-1).squeeze(-1)
            cls_velocity_t = self.scheduler._to_velocity(cls_model_output, cls_latents, t_cls, sample_eps)
            cls_velocity_next = self.scheduler._to_velocity(next_cls, euler_cls, t_next_cls, sample_eps)
            prev_cls = cls_latents + (t_next_cls - t_cls) * 0.5 * (cls_velocity_t + cls_velocity_next)
        return prev_latents, prev_cls

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        num_inference_steps: int = 25,
        guidance_scale: float = 3.7,
        guidance_interval: Tuple[float, float] = (0.0, 0.87),
        guidance_cls_scale: Optional[float] = None,
        guidance_cls_interval: Optional[Tuple[float, float]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[RiTPipelineOutput, Tuple]:
        class_labels_tensor = self._normalize_class_labels(class_labels)
        batch_size = class_labels_tensor.numel()
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        guidance_cls_scale = guidance_scale if guidance_cls_scale is None else guidance_cls_scale
        guidance_cls_interval = guidance_interval if guidance_cls_interval is None else guidance_cls_interval

        latents, cls_latents = self.prepare_latents(batch_size, generator=generator, dtype=dtype, device=device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for step_index in range(len(timesteps) - 1):
            timestep_value = timesteps[step_index]
            next_timestep_value = timesteps[step_index + 1]
            timestep = torch.full((batch_size,), float(timestep_value), device=device, dtype=dtype)
            next_timestep = torch.full((batch_size,), float(next_timestep_value), device=device, dtype=dtype)
            while timestep.ndim < latents.ndim:
                timestep = timestep.view(batch_size, *([1] * (latents.ndim - 1)))
                next_timestep = next_timestep.view(batch_size, *([1] * (latents.ndim - 1)))

            model_output, cls_output = self._predict_with_cfg(
                latents,
                timestep,
                class_labels_tensor,
                guidance_scale,
                guidance_interval,
                guidance_cls_scale,
                guidance_cls_interval,
                cls_latents=cls_latents,
            )

            if self.scheduler._cfg("sampling_method") == "heun" and step_index < len(timesteps) - 2:
                latents, cls_latents = self._heun_update(
                    latents,
                    model_output,
                    timestep,
                    next_timestep,
                    class_labels_tensor,
                    guidance_scale,
                    guidance_interval,
                    guidance_cls_scale,
                    guidance_cls_interval,
                    cls_latents=cls_latents,
                    cls_model_output=cls_output,
                )
            else:
                latents, cls_latents = self._euler_update(
                    latents,
                    model_output,
                    timestep,
                    next_timestep,
                    cls_latents=cls_latents,
                    cls_model_output=cls_output,
                )

        if output_type == "latent":
            if not return_dict:
                return (latents,)
            return RiTPipelineOutput(images=[], latents=latents)

        decoded = self.autoencoder.decode(latents, return_dict=True).sample
        images = self.image_processor.postprocess(decoded, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return RiTPipelineOutput(images=images, latents=latents)
