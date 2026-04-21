"""Linear shrinkage covariance estimator with a pluggable target."""

from enum import IntEnum

import numpy as np

from ..variance_predictor import VariancePredictor
from ._common import (
    correlation_from_covariance,
    schafer_strimmer_alpha,
    sample_covariance,
    single_index_covariance,
)


class Target(IntEnum):
    r"""Shrinkage target selector.

    The three targets surveyed in Pantaleo et al. (2010), Section III.D:

    - [`COMMON_COVARIANCE`][tradingflow.operators.predictors.variance.shrinkage.Target.COMMON_COVARIANCE]
      — diagonal = average sample variance, off-diagonal = average
      sample covariance.
    - [`CONSTANT_CORRELATION`][tradingflow.operators.predictors.variance.shrinkage.Target.CONSTANT_CORRELATION]
      — diagonal = sample variances, off-diagonal =
      \(\bar{r} \cdot \mathrm{std}_i \cdot \mathrm{std}_j\) with
      \(\bar{r}\) the average off-diagonal sample correlation.
    - [`SINGLE_INDEX`][tradingflow.operators.predictors.variance.shrinkage.Target.SINGLE_INDEX]
      — single-index factor-model covariance
      \(\sigma_f^{2} \beta \beta^T + \mathrm{diag}(\sigma_\epsilon^{2})\)
      using the equal-weighted cross-sectional mean as the market proxy.
    """

    COMMON_COVARIANCE = 1
    CONSTANT_CORRELATION = 2
    SINGLE_INDEX = 3


class Shrinkage(VariancePredictor[np.ndarray]):
    r"""Linear-shrinkage covariance estimator with a pluggable target.

    Computes \(\Sigma = \alpha F + (1 - \alpha) S\) where \(S\) is the
    sample covariance and \(F\) is one of the three structured targets
    enumerated by [`Target`][tradingflow.operators.predictors.variance.shrinkage.Target].

    The intensity \(\alpha\) is estimated analytically via the
    Schäfer-Strimmer (2005) element-wise unbiased estimator, as
    prescribed by Pantaleo et al. (2010).  Ignores features.

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
    target
        Shrinkage target, a member of
        [`Target`][tradingflow.operators.predictors.variance.shrinkage.Target].
        Default is `Target.COMMON_COVARIANCE`.
    verbose
        If `True`, print shrinkage diagnostics to stdout.
    **kwargs
        Forwarded to [`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor].
    """

    def __init__(
        self,
        universe,
        features_series,
        adjusted_prices_series,
        *,
        target: Target = Target.COMMON_COVARIANCE,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            features_series,
            adjusted_prices_series,
            fit_fn=lambda x, y: _fit_fn(y, target=target, verbose=verbose),
            predict_fn=lambda state, x, params: params,
            **kwargs,
        )


def _target_common_covariance(y: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, str]:
    N = S.shape[0]
    off_mask = ~np.eye(N, dtype=bool)
    avg_var = float(np.mean(np.diag(S)))
    avg_cov = float(S[off_mask].mean()) if N > 1 else avg_var
    F = np.full((N, N), avg_cov, dtype=np.float64)
    np.fill_diagonal(F, avg_var)
    return F, f"avg_var={avg_var:.4e}, avg_cov={avg_cov:.4e}"


def _target_constant_correlation(y: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, str]:
    N = S.shape[0]
    C, stds = correlation_from_covariance(S)
    off_mask = ~np.eye(N, dtype=bool)
    r_bar = float(C[off_mask].mean()) if N > 1 else 1.0
    F = r_bar * np.outer(stds, stds)
    np.fill_diagonal(F, np.diag(S))
    return F, f"r_bar={r_bar:.4f}"


def _target_single_index(y: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, str]:
    return single_index_covariance(y), ""


_TARGET_BUILDERS = {
    Target.COMMON_COVARIANCE: _target_common_covariance,
    Target.CONSTANT_CORRELATION: _target_constant_correlation,
    Target.SINGLE_INDEX: _target_single_index,
}


def _fit_fn(y: np.ndarray, *, target: Target, verbose: bool = False) -> np.ndarray:
    T, N = y.shape

    S, centered, finite = sample_covariance(y)
    F, diagnostics = _TARGET_BUILDERS[target](y, S)

    alpha, T_eff = schafer_strimmer_alpha(S, F, centered, finite)

    if verbose:
        extras = f", {diagnostics}" if diagnostics else ""
        print(f"  shrinkage[{target.value}]: {T_eff}/{T} valid samples, {N} stocks{extras}, alpha={alpha:.4f}")

    return alpha * F + (1.0 - alpha) * S
