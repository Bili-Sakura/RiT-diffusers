__all__ = [
    "RiTAutoencoderKL",
    "RiTTransformer2DModel",
    "RiTFlowMatchScheduler",
    "RiTPipeline",
    "RiTPipelineOutput",
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
    if name == "RiTAutoencoderKL":
        from .models.autoencoders import RiTAutoencoderKL

        return RiTAutoencoderKL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
