# Copyright 2026 The NYU Vision-X and HuggingFace Teams. All rights reserved.
#
# Vendored from Hugging Face Diffusers (commit e39aecff):
# https://github.com/huggingface/diffusers/blob/e39aecff57ed14d1018529c3de6ec3c34fadb559/src/diffusers/models/autoencoders/autoencoder_rae.py

from ._upstream import load_vendored_autoencoder_rae

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_vendored_autoencoder_rae()
    return _MODULE


def __getattr__(name: str):
    if name in {"AutoencoderRAE", "RAEDecoder", "RAEDecoderOutput"}:
        return getattr(_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AutoencoderRAE", "RAEDecoder", "RAEDecoderOutput"]
