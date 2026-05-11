# RiT: Vanilla Diffusion Transformers Are Enough in Representation Space

Official PyTorch implementation of:

> **RiT: Vanilla Diffusion Transformers Are Enough in Representation Space**
> Le Zhang, Ning Mang, Aishwarya Agrawal

RiT is a class-conditional image generator that performs flow matching in the
frozen DINOv2 representation space. A vanilla Diffusion Transformer, trained
with **x-prediction** on **element-wise standardized** features, a
**dimension-aware noise schedule**, and a **joint [CLS]-patch** objective, is
enough to reach state-of-the-art FID on ImageNet 256x256 — with no DDT head,
no Riemannian reformulation, and no representation-alignment loss.

---

## Results on ImageNet 256x256

RiT-XL uses the smallest DINOv2 variant (DINOv2-S, d=384) and 676M denoiser
parameters. All FIDs are measured with 25 Heun steps and classifier-free
guidance with the time-shift schedule.

| Method                   | Params | w/o guidance FID | w/ guidance FID |
|--------------------------|-------:|-----------------:|----------------:|
| DiT-XL  (SD-VAE)         |  675M  |      9.62        |      2.27       |
| SiT-XL  (SD-VAE)         |  675M  |      8.61        |      2.06       |
| REPA-XL (SD-VAE)         |  675M  |      5.78        |      1.29       |
| DDT-XL  (SD-VAE)         |  675M  |      6.27        |      1.26       |
| REG-XL  (SD-VAE)         |  675M  |      1.80        |      1.36       |
| RAE-XL  (DINOv2-S)       |  676M  |      1.87        |      1.41       |
| RAE-XL^DH (DINOv2-B)     |  839M  |      1.51        |      1.16       |
| FAE-XL  (DINOv2-G)       |  675M  |      1.48        |      1.29       |
| **RiT-XL (DINOv2-S, ours)** | **676M** | **1.45** | **1.14** |

RiT also supports few-step generation out of the box (no distillation, no
consistency training):

| Heun steps | 5  | 10  | 25  | 50  |
|-----------:|---:|----:|----:|----:|
| FID (CFG=1.0) | 2.44 | 1.59 | 1.47 | 1.46 |
| FID (CFG=3.7) | 1.99 | 1.27 | 1.15 | 1.15 |

---

## Installation

```bash
git clone https://github.com/lezhang7/RiT.git
cd RiT
pip install -r requirements.txt
```

Python 3.10+ and a CUDA-capable GPU are assumed. The DINOv2 encoder is loaded
automatically from HuggingFace.

### RAE decoder weights

The frozen decoder that maps DINOv2 features back to pixels is from
[RAE](https://github.com/bytetriper/RAE). Place the pretrained decoders under:

```
models/decoders/dinov2/wReg_small/ViTXL_n08/model.pt   # for DINOv2-Small
models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt    # for DINOv2-Base
```

See the [RAE repo](https://github.com/bytetriper/RAE) for download links.

### ImageNet data

Organize ImageNet as standard `ImageFolder`:

```
imagenet/
  train/
    n01440764/*.JPEG
    ...
  val/
    ...
```

### FID reference statistics

Evaluation expects `fid_stats/adm_in256_stats.npz` (and optionally
`fid_stats/adm_in512_stats.npz` for 512x512). This is the Inception FID
reference computed from the ADM ImageNet reference batch. Generate it with
`torch_fidelity` or download a published copy and place it under `fid_stats/`.


---

## Quick start

### 1. Compute normalization statistics (already shipped for DINOv2-S/B)

`stats/RAE_DINOv2_small/normalization_stats.pt` and
`stats/RAE_DINOv2_base/normalization_stats.pt` are included. To recompute:

```bash
bash scripts/calc_stats.sh dinov2s       # DINOv2-Small
bash scripts/calc_stats.sh dinov2b       # DINOv2-Base
```

### 2. Train RiT-XL (main recipe — reproduces FID 1.45)

```bash
bash scripts/train.sh
# or override paths:
OUTPUT_DIR=output/my_run IMAGENET_PATH=/data/imagenet bash scripts/train.sh
```

Runs on 8 GPUs, batch 192/GPU (effective 1536), 800 epochs. See
`scripts/train.sh` for the full command.

### 3. Evaluate a checkpoint (FID / IS / Precision / Recall)

```bash
# Guided (CFG=3.7 in [0.1, 0.98]):   FID ~1.14
CKPT=output/my_run/checkpoint-last.pth CFG=3.7 bash scripts/eval.sh

# Unguided:                            FID ~1.45
CKPT=output/my_run/checkpoint-last.pth CFG=1.0 bash scripts/eval.sh

# Few-step:                            10 Heun steps → FID ~1.27 guided
CKPT=output/my_run/checkpoint-last.pth NUM_STEPS=10 CFG=3.7 bash scripts/eval.sh
```

### 4. Reconstruction demo (encoder + decoder round-trip)

```bash
python sample_rae.py --image demo/pixabay_cat.png --rae_model RAE_DINOv2 --dinov2small
```

---

## Ablations (Table 3)

Each script varies one factor at a time:

```bash
bash scripts/ablate_vpred.sh           # v-prediction instead of x-prediction
bash scripts/ablate_logit_normal.sh    # JiT logit-normal noise schedule (no time shift)
bash scripts/ablate_nocls.sh           # no joint [CLS]-patch modeling
bash scripts/ablate_dinov2b.sh         # DINOv2-Base encoder (d=768 instead of 384)
```

> Note: training without element-wise standardization diverges (FID > 300
> throughout), so it is the default behavior — remove `--rae_normalize` at
> your own risk.

---

## Code layout

```
├── main.py              # Training / evaluation entry point
├── model.py             # RiT transformer backbone (adaLN, SwiGLU, RMSNorm, RoPE)
├── denoiser.py          # Flow-matching loss, time schedules, ODE samplers, CFG
├── engine.py            # Training loop, FID / IS / P-R computation
├── rae.py               # Frozen DINOv2 encoder + ViT-MAE-style decoder
├── rae_decoder.py       # GeneralDecoder implementation
├── calculate_stat.py    # Precompute normalization statistics
├── sample_rae.py        # Encoder-decoder reconstruction demo
├── decoder_config/      # DINOv2-S/B decoder configs (local)
├── stats/               # Precomputed normalization statistics
├── scripts/             # train / eval / ablation shell scripts
└── util/
    ├── crop.py          # Center-cropping (ADM)
    ├── lr_sched.py      # Constant + cosine LR schedules
    ├── model_util.py    # VisionRoPE, sin-cos pos embed, RMSNorm
    ├── muon.py          # Muon optimizer param split (optional)
    ├── mae_utils.py     # ViTMAEConfig helpers for GeneralDecoder
    └── misc.py          # Distributed training, checkpointing, logging
```

### Model variants

All variants use SwiGLU, QK-normalized attention, 2D VisionRoPE, and 32
in-context tokens; differences are in depth/width.

| `--model`      | Layers | Hidden | Heads | FFN   | Params |
|----------------|-------:|-------:|------:|------:|-------:|
| `RiT-S/16`     |   12   |   384  |   6   | 1536  |  ~33M  |
| `RiT-B/16`     |   12   |   768  |  12   | 3072  | ~130M  |
| `RiT-L/16`     |   24   |  1024  |  16   | 4096  | ~458M  |
| `RiT-XL/16`    |   28   |  1152  |  16   | 4608  |  676M  |
| `RiT-H/16`     |   32   |  1280  |  16   | 5120  | ~900M  |

---

## Citation

If you find RiT useful, please cite:

```bibtex
@article{zhang2025rit,
  title  = {RiT: Vanilla Diffusion Transformers Are Enough in Representation Space},
  author = {Zhang, Le and Mang, Ning and Agrawal, Aishwarya},
  year   = {2025}
}
```

---

## Acknowledgments

This codebase is built on top of two projects whose ideas and implementations
it directly extends:

- **JiT** — [LTH14/JiT](https://github.com/LTH14/JiT): x-prediction flow
  matching in pixel space, in-context class tokens, and the modernized DiT
  block design (SwiGLU, RMSNorm, QK-norm, RoPE) that we adopt verbatim.
- **RAE** — [bytetriper/RAE](https://github.com/bytetriper/RAE): DINOv2
  encoder + ViT decoder setup for representation-space diffusion. We reuse
  their frozen encoder/decoder pairing.

We also thank the authors of DiT, SiT, LightningDiT, REPA, REG, and DDT whose
work informed the design choices of this project. See the paper for full
citations.
