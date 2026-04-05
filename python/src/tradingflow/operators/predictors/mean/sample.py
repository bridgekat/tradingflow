"""Sample mean predictor."""

import numpy as np

from ..mean_predictor import MeanPredictor, MeanPredictorState


class Sample(MeanPredictor[np.ndarray]):
    """Predict future returns as the sample mean of past returns.

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
    """Sample mean of returns per stock (NaN-robust). Ignores features."""
    # y: (T, N)
    return np.nanmean(y, axis=0)  # (N,)


def _predict_fn(state: MeanPredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Return the sample mean directly."""
    return params
