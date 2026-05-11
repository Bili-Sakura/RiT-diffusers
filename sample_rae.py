#!/usr/bin/env python3
"""
Run a stage-1 RAE reconstruction from a config file.
Saves original (resized), reconstruction, and side-by-side comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from rae import RAE_models

DEFAULT_IMAGE = Path("demo/pixabay_cat.png")
DEFAULT_OUTPUT = Path("./recon")


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct an input image using a Stage-1 RAE."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--rae_model", type=str, default="RAE_DINOv2")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dinov2small", action="store_true", default=False)
    args = parser.parse_args()

    device = get_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.rae_model

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    torch.set_grad_enabled(False)

    # Build RAE
    rae_kwargs = {}
    if args.rae_model == "RAE_DINOv2":
        rae_kwargs["dinov2small"] = args.dinov2small
    rae = RAE_models[args.rae_model](**rae_kwargs).to(device)
    rae.eval()

    # Load image
    img_pil = Image.open(args.image).convert("RGB")
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)  # (1, 3, H, W) [0,1]

    # Resize to encoder input size
    enc_size = rae.encoder_input_size
    img_resized = F.interpolate(img_tensor, size=(enc_size, enc_size), mode='bicubic', align_corners=False)

    print(f"Model: {model_tag}")
    print(f"Input: {args.image} {tuple(img_tensor.shape)} -> resized to {tuple(img_resized.shape)}")

    # Encode + Decode
    latent = rae.encode(img_resized)
    recon = rae.decode(latent).clamp(0, 1)

    # Metrics
    mse = F.mse_loss(recon.cpu(), img_resized.cpu()).item()
    import numpy as np
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    print(f"Latent shape: {tuple(latent.shape)}")
    print(f"PSNR: {psnr:.2f} dB")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_name = args.image.stem

    orig_path = args.output_dir / f"{img_name}_original_{enc_size}_{timestamp}.png"
    recon_path = args.output_dir / f"{img_name}_recon_{model_tag}_{timestamp}.png"
    comp_path = args.output_dir / f"{img_name}_comparison_{model_tag}_{timestamp}.png"

    save_image(img_resized, orig_path)
    save_image(recon, recon_path)
    comparison = torch.cat([img_resized, recon], dim=3)
    save_image(comparison, comp_path)

    print(f"\nSaved:")
    print(f"  Original:    {orig_path}")
    print(f"  Recon:       {recon_path}")
    print(f"  Comparison:  {comp_path}")


if __name__ == "__main__":
    main()
