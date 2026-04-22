"""Sample covariance predictor."""

import numpy as np

from ..variance_predictor import VariancePredictor, VariancePredictorState
from ._common import sample_covariance


class Sample(VariancePredictor[np.ndarray]):
    """Predict covariance as the sample covariance of the target series.

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
    target_series
        Recorded target series, element shape `(num_stocks,)`.  The
        covariance estimator operates cross-sectionally on this series.
    **kwargs
        Forwarded to [`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor].
    """

    def __init__(
        self,
        universe,
        features_series,
        target_series,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            features_series,
            target_series,
            fit_fn=_fit_fn,
            predict_fn=_predict_fn,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample covariance of target (NaN-robust). Ignores features."""
    S, _, _ = sample_covariance(y)
    return S


def _predict_fn(state: VariancePredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params
