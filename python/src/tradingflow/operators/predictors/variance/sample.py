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
    features
        Stacked features, shape ``(num_stocks, num_features)``.
        Passed through but not used.
    adjusted_prices
        Stacked forward-adjusted close prices, shape ``(num_stocks,)``.
    rebalance_period
        Produce output every N ticks.
    max_samples
        Maximum number of time rows.
    """

    def __init__(
        self,
        universe,
        features,
        adjusted_prices,
        *,
        rebalance_period: int,
        max_samples: int,
    ) -> None:
        super().__init__(
            universe,
            features,
            adjusted_prices,
            fit_fn=_fit_fn,
            predict_fn=_predict_fn,
            rebalance_period=rebalance_period,
            max_samples=max_samples,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample covariance of returns (NaN-robust). Ignores features."""
    # y: (T, N)
    T, N = y.shape
    if T < 2:
        return np.eye(N)
    # Per-stock mean ignoring NaN.
    mean = np.nanmean(y, axis=0)
    centered = y - mean
    # Fill NaN with 0 so they contribute nothing to sums.
    finite = np.isfinite(centered)
    centered = np.where(finite, centered, 0.0)
    # Pairwise valid observation counts.
    indicator = finite.astype(np.float64)
    counts = indicator.T @ indicator  # (N, N)
    return (centered.T @ centered) / np.maximum(counts - 1, 1.0)


def _predict_fn(state: VariancePredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Return the sample covariance directly."""
    return params
