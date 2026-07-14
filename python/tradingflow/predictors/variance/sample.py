"""Sample covariance predictor (pyhost contract)."""

import numpy as np

from ._base import VariancePredictor, VariancePredictorState
from ._common import sample_covariance


class Sample(VariancePredictor[np.ndarray]):
    """Predict covariance as the sample covariance of the target series.

    Corresponds to the *Markowitz* direct estimator in Pantaleo et al.
    (2010).  NaN-robust via pairwise complete observations.  Ignores
    features.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
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


def build(**kwargs) -> Sample:
    """Construct a :class:`Sample` covariance predictor.

    Build kwargs
    ------------
    num_stocks : int
    num_features : int
    universe_size : int
    target_offset : int
    refit_every : int, optional (default 1)
    max_periods : int | None, optional
    min_periods : int | None, optional
    """
    return Sample(**kwargs)
