"""RiT extensions merged with the upstream Hugging Face Diffusers package."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # noqa: F401

try:
    from importlib.metadata import version as _package_version

    __version__ = _package_version("diffusers")
except Exception:
    __version__ = "0.38.0"

__all__ = [
    "AutoencoderRAE",
    "RiTTransformer2DModel",
    "RiTFlowMatchScheduler",
    "RiTPipeline",
    "RiTPipelineOutput",
    "create_rit_autoencoder",
]


def __getattr__(name: str):
    if name in {"RiTTransformer2DModel", "RiTTransformer2DModelOutput"}:
        from .models.transformers import RiTTransformer2DModel

        return RiTTransformer2DModel
    if name == "RiTFlowMatchScheduler":
        from .schedulers import RiTFlowMatchScheduler

        return RiTFlowMatchScheduler
    if name in {"RiTPipeline", "RiTPipelineOutput"}:
        from .pipelines import RiTPipeline, RiTPipelineOutput

        return {"RiTPipeline": RiTPipeline, "RiTPipelineOutput": RiTPipelineOutput}[name]
    if name == "AutoencoderRAE":
        from .models.autoencoders.autoencoder_rae import AutoencoderRAE

        return AutoencoderRAE
    if name == "create_rit_autoencoder":
        from .models.autoencoders.autoencoder_rae import create_rit_autoencoder

        return create_rit_autoencoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
