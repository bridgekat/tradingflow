"""Sample covariance predictor."""

import numpy as np

from ..variance_predictor import VariancePredictor, VariancePredictorState
from ._common import sample_covariance


class Sample(VariancePredictor[np.ndarray]):
    """Predict covariance as the sample covariance of past returns.

    Corresponds to the *Markowitz* direct estimator in Pantaleo et al.
    (2010).  NaN-robust via pairwise complete observations; see
    [`sample_covariance`][tradingflow.operators.predictors.variance._common.sample_covariance].
    Ignores features.

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
    S, _, _ = sample_covariance(y)
    return S


def _predict_fn(state: VariancePredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params
