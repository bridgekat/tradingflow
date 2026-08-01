from ._incremental import IncrementalMeanPredictor


def build(*, alpha: float = 1.0, **kwargs) -> IncrementalMeanPredictor:
    """Constructs an incremental pooled Ridge mean predictor.

    The penalty matches `ridge`: the sample-size-invariant
    `(1/n)‖·‖² + α‖β‖²` reduces to the normal equations `(ZᵀZ + α n I) β = Zᵀy`
    on the pool-standardized design. See
    `tradingflow.predictor.mean._incremental` for the design.
    """
    return IncrementalMeanPredictor(alpha=alpha, **kwargs)
