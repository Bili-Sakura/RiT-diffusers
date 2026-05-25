__all__ = [
    "AutoencoderRAE",
    "create_rit_autoencoder",
    "RiTTransformer2DModel",
    "RiTTransformer2DModelOutput",
]


def __getattr__(name: str):
    if name in {"RiTTransformer2DModel", "RiTTransformer2DModelOutput"}:
        from .transformers import RiTTransformer2DModel, RiTTransformer2DModelOutput

        return {"RiTTransformer2DModel": RiTTransformer2DModel, "RiTTransformer2DModelOutput": RiTTransformer2DModelOutput}[
            name
        ]
    if name in {"AutoencoderRAE", "create_rit_autoencoder"}:
        from .autoencoders import AutoencoderRAE, create_rit_autoencoder

        return {"AutoencoderRAE": AutoencoderRAE, "create_rit_autoencoder": create_rit_autoencoder}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
