"""Sample covariance predictor."""

import numpy as np

from ..variance_predictor import VariancePredictor, VariancePredictorState


class Sample(VariancePredictor[np.ndarray]):
    """Predict covariance as the sample covariance of past returns.

    Ignores features entirely.  Useful as a baseline.

    Parameters
    ----------
    universe
        Universe weights, shape ``(num_stocks,)``.
    features_series
        Recorded features series, element shape ``(num_stocks, num_features)``.
        Passed through but not used.
    adjusted_prices_series
        Recorded forward-adjusted close prices series, element shape
        ``(num_stocks,)``.
    **kwargs
        Forwarded to [`VariancePredictor`][tradingflow.operators.predictors.VariancePredictor].
    """

    def __init__(
        self,
        universe,
        features_series,
        adjusted_prices_series,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            features_series,
            adjusted_prices_series,
            fit_fn=_fit_fn,
            predict_fn=_predict_fn,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample covariance of returns (NaN-robust). Ignores features."""
    # y: (T, N)
    T, N = y.shape

    # NaN-robust sample covariance (pairwise complete observations).
    mean = np.nanmean(y, axis=0)
    centered = y - mean
    finite = np.isfinite(centered)
    centered = np.where(finite, centered, 0.0)
    indicator = finite.astype(np.float64)
    counts = indicator.T @ indicator  # (N, N)
    return (centered.T @ centered) / np.maximum(counts - 1, 1.0)


def _predict_fn(state: VariancePredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Return the sample covariance directly."""
    return params
