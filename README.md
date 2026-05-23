<div align="center">

# RiT: Vanilla Diffusion Transformers Suffice in Representation Space

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2605.21981) &nbsp;
[![Checkpoint](https://img.shields.io/badge/🤗%20Checkpoint-le723z%2FRiT-FFD21E.svg)](https://huggingface.co/le723z/RiT) &nbsp;
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**[Le Zhang](https://lezhang7.github.io/) &nbsp;·&nbsp; Ning Mang &nbsp;·&nbsp; Aishwarya Agrawal**

<img src="assets/samples_main.png" width="92%" alt="RiT samples on ImageNet 256x256"/>

</div>

RiT is a vanilla Diffusion Transformer trained directly in the **384-dim
DINOv2 feature space** — no DDT heads, Riemannian reformulations, or REPA
losses. The recipe: x-prediction on element-wise standardized features, a
dimension-aware noise schedule, and a joint [CLS]-patch objective. See the
[paper](https://arxiv.org/pdf/2605.21981) for the full method and analysis.

<div align="center">
<img src="assets/method.png" width="55%" alt="RiT architecture"/>
</div>

## Results on ImageNet 256×256

RiT-XL, 25 Heun steps with the time-shift schedule.

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

## Quick start

### Install

```bash
git clone https://github.com/lezhang7/RiT.git
cd RiT
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
```

ImageNet should follow the standard `ImageFolder` layout
(`imagenet/train/n01440764/*.JPEG`). Normalization stats and Inception FID
reference stats are already in `stats/` and `fid_stats/`. The DINOv2 encoder,
RAE decoder, and RiT-XL checkpoint are fetched from HuggingFace on first use;
to prefetch:

```bash
python scripts/download_assets.py
```

### Train

```bash
bash scripts/train.sh
# override paths if needed:
OUTPUT_DIR=output/my_run IMAGENET_PATH=/data/imagenet bash scripts/train.sh
```

8 GPUs · batch 192/GPU · 800 epochs.

### Evaluate

```bash
bash scripts/eval.sh                                # CFG=3.7,   FID ~1.14
CFG=1.0 bash scripts/eval.sh                        # unguided,  FID ~1.45
NUM_STEPS=10 bash scripts/eval.sh                   # 10 steps,  FID ~1.27
CKPT=output/my_run/checkpoint-last.pth bash scripts/eval.sh
COMPUTE_PRC=1 bash scripts/eval.sh                  # +Precision/Recall (~1.5 GB DL)
```

## Citation

```bibtex
@article{zhang2026rit,
  title   = {RiT: Vanilla Diffusion Transformers Suffice in Representation Space},
  author  = {Zhang, Le and Mang, Ning and Agrawal, Aishwarya},
  journal = {arXiv preprint arXiv:2605.21981},
  year    = {2026}
}
```

## Acknowledgments

Built on [JiT](https://github.com/LTH14/JiT) (x-prediction flow matching, modernized DiT blocks)
and [RAE](https://github.com/bytetriper/RAE) (DINOv2 encoder + ViT decoder).
