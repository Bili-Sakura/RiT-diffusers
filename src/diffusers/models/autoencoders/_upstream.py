# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Load vendored AutoencoderRAE with upstream Diffusers dependencies."""

from __future__ import annotations

import importlib
import importlib.util
import site
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator, Optional

_LOCAL_SRC = Path(__file__).resolve().parents[3]
_VENDORED_RAE = Path(__file__).with_name("autoencoder_rae_upstream.py")


def _find_upstream_diffusers_root() -> Optional[Path]:
    for entry in site.getsitepackages() + ([site.getusersitepackages()] if site.getusersitepackages() else []):
        root = Path(entry) / "diffusers"
        if (root / "configuration_utils.py").exists():
            return root
    return None


@contextmanager
def _without_local_overlay() -> Iterator[None]:
    original_path = list(sys.path)
    overlay_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "diffusers" or name.startswith("diffusers.")
    }
    filtered_path = [entry for entry in original_path if Path(entry).resolve() != _LOCAL_SRC]
    try:
        for name in overlay_modules:
            del sys.modules[name]
        sys.path = filtered_path
        yield
    finally:
        sys.path = original_path
        for name, module in overlay_modules.items():
            sys.modules[name] = module


def load_vendored_autoencoder_rae() -> ModuleType:
    upstream_root = _find_upstream_diffusers_root()
    if upstream_root is None:
        raise ImportError("Install diffusers>=0.38.0 to use AutoencoderRAE.")

    module_name = "diffusers.models.autoencoders.autoencoder_rae"
    with _without_local_overlay():
        importlib.import_module("diffusers")
        spec = importlib.util.spec_from_file_location(
            module_name,
            _VENDORED_RAE,
            submodule_search_locations=[str(upstream_root), str(upstream_root / "models")],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load vendored module at {_VENDORED_RAE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
