from .granularity_predictor import HitGranularityPredictor
from .pipeline import GranuScore

def granuscore(
    texts,
    **kwargs,
):
    return GranuScore()(
        texts,
        **kwargs,
    )

__all__ = ["GranuScore", "granuscore", "HitGranularityPredictor"]
