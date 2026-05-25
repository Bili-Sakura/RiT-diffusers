<div align="center">

# RiT-diffusers

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21981) &nbsp;
[![Checkpoint](https://img.shields.io/badge/🤗%20Checkpoint-le723z%2FRiT-FFD21E.svg)](https://huggingface.co/le723z/RiT) &nbsp;
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Diffusers-native integration for [RiT: Vanilla Diffusion Transformers Suffice in Representation Space](https://arxiv.org/pdf/2605.21981).**

</div>

This repository mirrors the layout used by [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers.git): RiT components live under `src/diffusers` as `ModelMixin` / `ConfigMixin` classes, a custom flow-matching scheduler, and a `RiTPipeline` for end-to-end sampling. The legacy standalone training stack (`main.py`, `engine.py`, `denoiser.py`, `util/`, etc.) has been removed.

## Package layout

- `src/diffusers/models/transformers/transformer_rit.py` — `RiTTransformer2DModel`
- `src/diffusers/models/autoencoders/autoencoder_rae_upstream.py` — upstream `AutoencoderRAE` implementation ([Diffusers e39aecff](https://github.com/huggingface/diffusers/blob/e39aecff57ed14d1018529c3de6ec3c34fadb559/src/diffusers/models/autoencoders/autoencoder_rae.py))
- `src/diffusers/models/autoencoders/autoencoder_rae_presets.py` — RiT presets (`create_rit_autoencoder`) and checkpoint loading
- `src/diffusers/schedulers/scheduling_flow_match_rit.py` — `RiTFlowMatchScheduler` (x-prediction, time-shift, Heun/Euler)
- `src/diffusers/pipelines/rit/pipeline_rit.py` — `RiTPipeline` with classifier-free guidance
- `scripts/convert_rit_to_diffusers.py` — convert original RiT checkpoints
- `scripts/sample_rit.py` — sample from a converted pipeline

## Install

```bash
pip install -e .
# PyTorch must match your CUDA build, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Download assets

```bash
python scripts/download_assets.py
```

This fetches the RiT-XL checkpoint and DINOv2-Small RAE decoder weights expected by the conversion script.

## Convert a checkpoint

```bash
python scripts/convert_rit_to_diffusers.py \
  --checkpoint output/rit_xl_dinov2s/checkpoint-last.pth \
  --output rit-xl-diffusers \
  --model-size rit-xl \
  --dinov2-small \
  --rae-normalize \
  --check-load
```

The output directory contains:

```text
model_index.json
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
scheduler/scheduler_config.json
autoencoder/config.json
```

The conversion script writes a full `autoencoder/` folder via `AutoencoderRAE.save_pretrained`, including decoder weights and optional latent statistics.

## Sample

```bash
PYTHONPATH=src python scripts/sample_rit.py \
  --model rit-xl-diffusers \
  --class-label 207 \
  --num-inference-steps 25 \
  --guidance-scale 3.7 \
  --guidance-low 0.0 \
  --guidance-high 0.87 \
  --seed 42
```

Programmatic usage:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from diffusers import RiTPipeline
import torch

pipe = RiTPipeline.from_pretrained("rit-xl-diffusers", torch_dtype=torch.bfloat16).to("cuda")
image = pipe(class_labels=207, num_inference_steps=25, guidance_scale=3.7).images[0]
image.save("sample.png")
```

## Upstreaming to Diffusers

Copy the files under `src/diffusers` into the matching locations in `huggingface/diffusers` and register the classes in Diffusers' lazy import tables. Module names and save/load artifacts follow Diffusers conventions.

## Citation

```bibtex
@article{zhang2026rit,
  title   = {RiT: Vanilla Diffusion Transformers Suffice in Representation Space},
  author  = {Zhang, Le and Mang, Ning and Agrawal, Aishwarya},
  journal = {arXiv preprint arXiv:2605.21981},
  year    = {2026}
}
```

## License

MIT
