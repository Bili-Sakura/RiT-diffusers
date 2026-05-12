#!/usr/bin/env python3
"""One-off uploader for the released RiT-XL checkpoint.

Used to publish `checkpoint-740.pth` to https://huggingface.co/le723z/RiT.
Re-running is safe: HuggingFace upload is content-addressed and skips
identical blobs.

Prereq:
    1. `pip install huggingface_hub`
    2. Authenticate via either:
         - `hf auth login` and follow the prompt, OR
         - export HF_TOKEN=hf_...   (write-scoped token from huggingface.co/settings/tokens)
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("ERROR: huggingface_hub not installed.", file=sys.stderr)
    sys.exit(1)


REPO_ID = "le723z/RiT"
CKPT_LOCAL = Path("checkpoint-740.pth")
CKPT_REMOTE = "checkpoint-740.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=CKPT_LOCAL,
                        help=f"Local checkpoint path (default: {CKPT_LOCAL})")
    parser.add_argument("--repo", default=REPO_ID,
                        help=f"HuggingFace repo (default: {REPO_ID})")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without uploading")
    args = parser.parse_args()

    ckpt = args.ckpt.resolve()
    if not ckpt.exists():
        print(f"ERROR: {ckpt} not found", file=sys.stderr)
        sys.exit(1)

    size_gb = ckpt.stat().st_size / 1e9
    print(f"Uploading {ckpt} ({size_gb:.2f} GB) -> https://huggingface.co/{args.repo}/blob/main/{CKPT_REMOTE}")

    if args.dry_run:
        print("[dry-run] skipping create_repo + upload_file")
        return

    api = HfApi()
    create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=CKPT_REMOTE,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload RiT-XL checkpoint (epoch 740, FID 1.45 / 1.14 on ImageNet 256x256)",
    )
    print(f"Done. Available at https://huggingface.co/{args.repo}/blob/main/{CKPT_REMOTE}")


if __name__ == "__main__":
    main()
