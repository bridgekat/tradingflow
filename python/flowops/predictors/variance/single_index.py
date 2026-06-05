"""Single-index factor model covariance predictor (flowops host contract)."""

import numpy as np

from ._base import VariancePredictor, VariancePredictorState
from ._common import single_index_covariance


class SingleIndex(VariancePredictor[np.ndarray]):
    r"""Single-index model covariance estimator.

    Fits \(r_i(t) = \alpha_i + \beta_i f(t) + \epsilon_i(t)\) stock-by-stock
    against the equal-weighted cross-sectional mean return (market-factor
    proxy) and returns
    \(\Sigma = \sigma_f^{2}\beta\beta^T + \mathrm{diag}(\sigma_\epsilon^{2})\).
    Corresponds to the *SI* estimator of Pantaleo et al. (2010).  Ignores
    features.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            fit_fn=_fit_fn,
            predict_fn=_predict_fn,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a single-index covariance from target matrix `y`."""
    return single_index_covariance(y)


def _predict_fn(state: VariancePredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params


def build(**kwargs) -> SingleIndex:
    """Construct a :class:`SingleIndex` covariance predictor.

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
    return SingleIndex(**kwargs)
