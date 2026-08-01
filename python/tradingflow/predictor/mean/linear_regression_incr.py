from ._incremental import IncrementalMeanPredictor


def build(**kwargs) -> IncrementalMeanPredictor:
    """Constructs an incremental pooled OLS mean predictor — Ridge at `alpha=0`.

    Fits the same model as `linear_regression` but from an incrementally
    maintained Gram, so the per-rebalance cost is independent of history
    length. See `tradingflow.predictor.mean._incremental` for the design.
    """
    assert "alpha" not in kwargs, "OLS is unpenalized; use ridge_incr for alpha > 0"
    return IncrementalMeanPredictor(alpha=0.0, **kwargs)
