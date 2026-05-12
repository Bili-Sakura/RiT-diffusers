#!/usr/bin/env python3
"""Download the RiT-XL checkpoint and the matching RAE decoder weights.

Both come from public HuggingFace repos and are placed at the paths the
training/eval scripts expect by default:

    models/decoders/dinov2/wReg_small/ViTXL_n08/model.pt   # RAE decoder
    output/rit_xl_dinov2s/checkpoint-last.pth              # RiT-XL ckpt

The script is idempotent: existing files are skipped. Run it directly, or
let `scripts/eval.sh` invoke it for you.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"ERROR: huggingface_hub not importable from {sys.executable}", file=sys.stderr)
    print(f"  underlying error: {e}", file=sys.stderr)
    print(f"  fix: `{sys.executable} -m pip install huggingface_hub`", file=sys.stderr)
    print(f"  or:  PYTHON=/path/to/venv/bin/python bash scripts/eval.sh", file=sys.stderr)
    sys.exit(1)


# (HF repo, filename in repo, local destination)
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
    # Symlink would be more efficient but breaks if the HF cache moves; copy
    # is the safer default for a release script.
    shutil.copy(cached, dest_path)
    print(f"[done]  {asset_key:20s} -> {dest_path}  ({dest_path.stat().st_size / 1e9:.2f} GB)")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", nargs="+", default=["rae_decoder_small", "rit_xl_ckpt"],
        choices=list(ASSETS.keys()),
        help="Which assets to download (default: RiT-XL checkpoint + DINOv2-Small decoder)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    for key in args.assets:
        fetch(key, force=args.force)


if __name__ == "__main__":
    main()
