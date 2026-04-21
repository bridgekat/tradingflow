"""Sample mean predictor."""

import numpy as np

from ..mean_predictor import MeanPredictor, MeanPredictorState


class Sample(MeanPredictor[np.ndarray]):
    """Predict future returns as the sample mean of past returns.

    Ignores features entirely.  Useful as a baseline.

    Parameters
    ----------
    universe
        Universe weights, shape `(num_stocks,)`.
    features_series
        Recorded features series, element shape `(num_stocks, num_features)`.
        Passed through but not used.
    adjusted_prices_series
        Recorded forward-adjusted close prices series, element shape
        `(num_stocks,)`.
    **kwargs
        Forwarded to [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor].
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
    """Sample mean of returns per stock (NaN-robust). Ignores features."""
    # y: (T, N)
    return np.nanmean(y, axis=0)  # (N,)


def _predict_fn(state: MeanPredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Return the sample mean directly."""
    return params
