#!/usr/bin/env python3
"""Download the RiT-XL checkpoint and matching RAE decoder weights for conversion."""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as error:
    print(f"ERROR: huggingface_hub is required: {error}", file=sys.stderr)
    sys.exit(1)


ASSETS = {
    "rae_decoder_small": (
        "nyu-visionx/RAE-collections",
        "decoders/dinov2/wReg_small/ViTXL_n08/model.pt",
        "models/decoders/dinov2/wReg_small/ViTXL_n08/model.pt",
    ),
    "rae_decoder_base": (
        "nyu-visionx/RAE-collections",
        "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
    ),
    "rit_xl_ckpt": (
        "le723z/RiT",
        "checkpoint-last.pth",
        "output/rit_xl_dinov2s/checkpoint-last.pth",
    ),
}


def fetch(asset_key: str, force: bool = False) -> Path:
    repo_id, filename, dest = ASSETS[asset_key]
    dest_path = Path(dest)
    if dest_path.exists() and not force:
        print(f"[skip] {asset_key:20s} -> {dest_path} (exists)")
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {asset_key:20s} from {repo_id}/{filename}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy(cached, dest_path)
    print(f"[done]  {asset_key:20s} -> {dest_path}  ({dest_path.stat().st_size / 1e9:.2f} GB)")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["rae_decoder_small", "rit_xl_ckpt"],
        choices=list(ASSETS.keys()),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for key in args.assets:
        fetch(key, force=args.force)


if __name__ == "__main__":
    main()
