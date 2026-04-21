"""Single-index factor model covariance predictor."""

import numpy as np

from ..variance_predictor import VariancePredictor, VariancePredictorState
from ._common import single_index_covariance


class SingleIndex(VariancePredictor[np.ndarray]):
    r"""Single-index model covariance estimator.

    Fits the factor model
    \(r_i(t) = \alpha_i + \beta_i f(t) + \epsilon_i(t)\) stock-by-stock
    against an equal-weighted cross-sectional mean return used as a
    proxy for the market factor \(f(t)\).  The estimated covariance
    matrix is then

    \[
    \Sigma = \sigma_f^{2} \beta \beta^T + \mathrm{diag}(\sigma_\epsilon^{2}),
    \]

    where \(\sigma_f^{2}\) is the market-factor variance and
    \(\sigma_\epsilon^{2}\) is the vector of idiosyncratic residual
    variances.

    Corresponds to the *SI* estimator of Pantaleo et al. (2010).
    Since TradingFlow does not take an external index as input, the
    cross-sectional mean return at each timestep serves as \(f(t)\).
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
        Forwarded to [`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor].
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
    """Fit a single-index covariance from return matrix `y`."""
    return single_index_covariance(y)


def _predict_fn(state: VariancePredictorState[np.ndarray], features: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params
