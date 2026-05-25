__all__ = [
    "RiTAutoencoderKL",
    "RiTAutoencoderOutput",
    "RiTTransformer2DModel",
    "RiTTransformer2DModelOutput",
]


def __getattr__(name: str):
    if name in {"RiTTransformer2DModel", "RiTTransformer2DModelOutput"}:
        from .transformers import RiTTransformer2DModel, RiTTransformer2DModelOutput

        return {"RiTTransformer2DModel": RiTTransformer2DModel, "RiTTransformer2DModelOutput": RiTTransformer2DModelOutput}[
            name
        ]
    if name in {"RiTAutoencoderKL", "RiTAutoencoderOutput"}:
        from .autoencoders import RiTAutoencoderKL, RiTAutoencoderOutput

        return {"RiTAutoencoderKL": RiTAutoencoderKL, "RiTAutoencoderOutput": RiTAutoencoderOutput}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
