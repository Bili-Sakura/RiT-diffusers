<div align="center">

# RiT: Vanilla Diffusion Transformers Suffice in Representation Space

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21981) &nbsp;
[![Checkpoint](https://img.shields.io/badge/🤗%20Checkpoint-le723z%2FRiT-FFD21E.svg)](https://huggingface.co/le723z/RiT) &nbsp;
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**[Le Zhang](https://lezhang7.github.io/) &nbsp;·&nbsp; Ning Mang &nbsp;·&nbsp; Aishwarya Agrawal**

<p>
<a href="https://mila.quebec">Mila — Québec AI Institute, UdeM</a> &nbsp;·&nbsp;
Utrecht University &nbsp;·&nbsp; Canada CIFAR AI Chair
</p>

<img src="assets/samples_main.png" width="92%" alt="RiT samples on ImageNet 256x256"/>

*Curated RiT-XL samples on ImageNet 256×256.*

</div>

---

## What is RiT?

RiT is a vanilla Diffusion Transformer that **effectively models distributions
in high-dimensional representation spaces**. Prior latent-diffusion work
compresses images to low-dim spaces first — SD-VAE uses 4-dim latents, FAE
bottlenecks to 32 — because diffusion was believed to need low dimensionality
to be tractable. RiT flows directly in the native **384-dim** DINOv2 feature
space, yet matches or beats every prior method at the same ImageNet FID
without DDT heads, Riemannian reformulations, or representation-alignment
losses.

The recipe is deliberately minimal: **x-prediction** on **element-wise
standardized** DINOv2 features, a **dimension-aware noise schedule** that
compensates for the per-token dimensionality, and a **joint [CLS]-patch**
objective. That's the whole thing — a vanilla DiT is enough once the
representation geometry is right.

<div align="center">
<img src="assets/method.png" width="55%" alt="RiT architecture"/>
</div>

> **Architecture.** Frozen DINOv2 encoder → element-wise standardize → vanilla
> DiT trained with x-prediction → denormalize → frozen ViT decoder. The
> projected [CLS] token is prepended to the patch sequence, attends jointly,
> and is predicted by a separate linear head.

---

## Why DINOv2 features are favorable for flow matching

Pixels and DINOv2 features sit on manifolds of nearly identical *intrinsic*
dimensionality (TwoNN gives `d̂ ≈ 33` for both), but DINOv2 embeds that
manifold far more favorably relative to the `N(0, I)` source of flow matching.
Measured on 10K ImageNet images:

| Geometric axis                     |   Pixel | SD-VAE | DINOv2 | DINOv2 vs Pixel |
|------------------------------------|--------:|-------:|-------:|----------------:|
| Intrinsic dim `d̂` (TwoNN)          |  33.6   | —      |  32.6  | ≈ same          |
| Effective rank (covariance)        |    45   |    98  |   327  | **7.3× higher** |
| Cov. condition number `κ(Σ_t=0.9)` | ≈ 2,000 | —      |  ≈ 56  | **35× better**  |
| Median per-coord. excess kurtosis  |  0.958  | 0.228  | 0.083  | **11.5× lower** |
| On-manifold interp. MSE            | 0.0136  | —      | 0.0080 | **1.7× lower**  |

SD-VAE is consistently intermediate, so the advantage comes from
representation-learning objectives, not mere compression. These four axes —
high effective rank, well-conditioned covariance, near-Gaussian marginals, and
on-manifold linear interpolants — jointly predict that the noise→data ODE on
DINOv2 should be smooth and **few-step solvable**, which is exactly what we
observe empirically (3.6× steeper truncation-error decay than pixel-space JiT).

---

## Results on ImageNet 256×256

RiT-XL uses the smallest DINOv2 variant (DINOv2-S, d=384) and 676M denoiser
parameters. All FIDs use 25 Heun steps with the time-shift schedule.

| Method                       | Encoder           | Dim  | Params |     FID ↓ (CFG=1) |     FID ↓ (CFG≈3.7) |
|------------------------------|------------------:|-----:|-------:|------------------:|--------------------:|
| DiT-XL                       | SD-VAE            |  4   |  675M  | 9.62              | 2.27                |
| SiT-XL                       | SD-VAE            |  4   |  675M  | 8.61              | 2.06                |
| REPA-XL                      | SD-VAE            |  4   |  675M  | 5.78              | 1.29                |
| DDT-XL                       | SD-VAE            |  4   |  675M  | 6.27              | 1.26                |
| REG-XL                       | SD-VAE            |  4   |  675M  | 1.80              | 1.36                |
| RAE-XL                       | DINOv2-S          | 384  |  676M  | 1.87              | 1.41                |
| RAE-XL<sup>DH</sup>          | DINOv2-B          | 768  |  839M  | 1.51              | 1.16                |
| FAE-XL                       | FAE-DINOv2-G      |  32  |  675M  | 1.48              | 1.29                |
| **RiT-XL (ours)**            | **DINOv2-S**      | **384** | **676M** | **1.45**     | **1.14**            |

**RiT is the only method in this table that denoises in a high-dimensional
representation space and still wins at FID.** Every other method either
operates on 4-dim SD-VAE latents or bottlenecks the encoder to ≤32 dim; RiT
runs flow matching natively on 384-dim DINOv2-S tokens with a vanilla DiT.

**Convergence is ~7× faster than RAE at matched encoder.** RiT-XL matches RAE's
800-epoch FID within ~100 epochs.

<div align="center">
<img src="assets/main_result.png" width="58%" alt="Convergence comparison"/>
</div>

**Few-step generation works out of the box** — no distillation, no consistency
training. With the time-shift schedule and coupled noise:

| Heun steps     |  5   | 10   | 25   | 50   |
|---------------:|-----:|-----:|-----:|-----:|
| FID (CFG=1.0)  | 2.38 | 1.58 | 1.45 | 1.44 |
| FID (CFG=3.7)  | 1.99 | 1.25 | 1.14 | 1.14 |

<div align="center">
<img src="assets/nfe_compare.png" width="44%" alt="Few-step FID at matched NFE"/> &nbsp;
<img src="assets/trajectory_curvature.png" width="44%" alt="ODE truncation error vs step count"/>
</div>

> *Left:* Few-step FID at matched NFE — RiT clears 2.0 at 5 NFE while pixel-space
> JiT is at 26.2.
> *Right:* RiT's pixel-space truncation error decays 3.6× steeper than JiT's,
> consistent with a smoother velocity field over DINOv2 features.

---

## Installation

```bash
git clone https://github.com/lezhang7/RiT.git
cd RiT
# Install PyTorch matching your CUDA toolkit (12.8 example below)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The DINOv2-with-Registers encoder is loaded from HuggingFace on first use.
The RiT-XL checkpoint and the matching RAE decoder are also pulled from
HuggingFace automatically the first time you run `bash scripts/eval.sh`
(see [Quick start](#quick-start)). To prefetch them manually:

```bash
python scripts/download_assets.py            # decoder + RiT-XL ckpt
python scripts/download_assets.py --assets rae_decoder_base   # also DINOv2-Base decoder
```

This populates:
- `models/decoders/dinov2/wReg_small/ViTXL_n08/model.pt` — RAE decoder, from
  [`nyu-visionx/RAE-collections`](https://huggingface.co/nyu-visionx/RAE-collections).
- `output/rit_xl_dinov2s/checkpoint-last.pth` — released RiT-XL weights, from
  [`le723z/RiT`](https://huggingface.co/le723z/RiT).

### ImageNet data

Standard `ImageFolder` layout:

```
imagenet/
  train/n01440764/*.JPEG
  val/...
```

### What's already in the repo

- `stats/RAE_DINOv2_{small,base}/normalization_stats.pt` — element-wise
  standardization statistics for §3.1.
- `fid_stats/imagenet{256,512}_stats.npz` — Inception FID reference statistics
  used by `torch_fidelity` at evaluation time. No additional downloads needed.

---

## Quick start

### Train RiT-XL (reproduces FID 1.45 / 1.14)

```bash
bash scripts/train.sh
# or override paths:
OUTPUT_DIR=output/my_run IMAGENET_PATH=/data/imagenet bash scripts/train.sh
```

8 GPUs · batch 192/GPU (effective 1536) · 800 epochs. See `scripts/train.sh`
for the full command.

### Evaluate the released checkpoint

The first run downloads `le723z/RiT/checkpoint-last.pth` (RiT-XL, 800 epochs)
and the matching RAE decoder automatically — no manual setup:

```bash
# Guided (CFG=3.7 in [0.1, 0.98]):  FID ~1.14
bash scripts/eval.sh

# Unguided:                          FID ~1.45
CFG=1.0 bash scripts/eval.sh

# Few-step:                          10 Heun steps → FID ~1.27 guided
NUM_STEPS=10 bash scripts/eval.sh

# Evaluate your own checkpoint instead of the released one
CKPT=output/my_run/checkpoint-last.pth bash scripts/eval.sh
```

> **Note.** `bash scripts/eval.sh` runs in a fresh subshell and does **not**
> inherit `conda activate` from your terminal. If you hit
> `huggingface_hub not importable`, either activate your environment first or
> point the script at the right interpreter:
> ```bash
> PYTHON=/path/to/venv/bin/python TORCHRUN=/path/to/venv/bin/torchrun \
>     bash scripts/eval.sh
> ```

> **Wandb.** Both `train.sh` and `eval.sh` set `WANDB_MODE=disabled` by
> default so the scripts run out of the box without any wandb account setup.
> FID / IS / Precision / Recall are printed to stdout and appended to
> `${OUTPUT_DIR}/fid_results.txt` regardless. To log a run to wandb instead,
> override before invoking:
> ```bash
> export WANDB_MODE=online WANDB_API_KEY=... WANDB_PROJECT=RiT
> bash scripts/train.sh
> ```

Reports FID and Inception Score via [torch-fidelity](https://github.com/LTH14/torch-fidelity)
using the shipped `fid_stats/imagenet256_stats.npz` (Inception mean/cov on
ImageNet). To also compute **Precision / Recall** you need the ADM
reference-image batch (~1.5 GB, downloaded automatically on first use):

```bash
# Compute FID + IS + Precision + Recall
COMPUTE_PRC=1 bash scripts/eval.sh
```

### Reconstruction demo

```bash
python sample_rae.py --image demo/pixabay_cat.png --rae_model RAE_DINOv2 --dinov2small
```

---

## Code layout

```
├── main.py              # Training / evaluation entry point
├── model.py             # RiT transformer backbone (adaLN, SwiGLU, RMSNorm, RoPE)
├── denoiser.py          # Flow-matching loss, time schedules, ODE samplers, CFG
├── engine.py            # Training loop, FID / IS / P-R computation
├── rae.py               # Frozen DINOv2 encoder + ViT decoder
├── rae_decoder.py       # GeneralDecoder implementation
├── calculate_stat.py    # Precompute normalization statistics
├── sample_rae.py        # Encoder-decoder reconstruction demo
├── decoder_config/      # DINOv2-S/B decoder configs (local)
├── stats/               # Precomputed normalization statistics
├── fid_stats/           # Inception FID reference statistics
├── scripts/             # train / eval / calc_stats shell scripts
└── util/
    ├── crop.py          # Center-cropping (ADM)
    ├── lr_sched.py      # Constant + cosine LR schedules
    ├── model_util.py    # VisionRoPE, sin-cos pos embed, RMSNorm
    ├── mae_utils.py     # ViTMAEConfig helpers for the decoder
    └── misc.py          # Distributed training, checkpointing, logging
```

---

## Citation

If you find RiT useful, please cite:

```bibtex
@article{zhang2026rit,
  title   = {RiT: Vanilla Diffusion Transformers Suffice in Representation Space},
  author  = {Zhang, Le and Mang, Ning and Agrawal, Aishwarya},
  journal = {arXiv preprint arXiv:2605.21981},
  year    = {2026}
}
```

---

## Acknowledgments

This codebase builds directly on:

- **JiT** — [LTH14/JiT](https://github.com/LTH14/JiT): x-prediction flow
  matching in pixel space, in-context class tokens, and the modernized DiT
  block design (SwiGLU, RMSNorm, QK-norm, RoPE).
- **RAE** — [bytetriper/RAE](https://github.com/bytetriper/RAE): the
  frozen DINOv2 encoder + ViT decoder pairing for representation-space
  diffusion.

We also thank the authors of DiT, SiT, LightningDiT, REPA, REG, DDT, and
torch-fidelity for the tooling and design choices we relied on. Full citations
are in the paper.
