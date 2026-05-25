__all__ = [
    "AutoencoderRAE",
    "RAEDecoder",
    "RAEDecoderOutput",
    "RIT_RAE_PRESETS",
    "create_rit_autoencoder",
]


def __getattr__(name: str):
    if name in {"AutoencoderRAE", "RAEDecoder", "RAEDecoderOutput"}:
        from .autoencoder_rae import AutoencoderRAE, RAEDecoder, RAEDecoderOutput

        return {"AutoencoderRAE": AutoencoderRAE, "RAEDecoder": RAEDecoder, "RAEDecoderOutput": RAEDecoderOutput}[name]
    if name in {"RIT_RAE_PRESETS", "create_rit_autoencoder"}:
        from .autoencoder_rae_presets import RIT_RAE_PRESETS, create_rit_autoencoder

        return {"RIT_RAE_PRESETS": RIT_RAE_PRESETS, "create_rit_autoencoder": create_rit_autoencoder}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
