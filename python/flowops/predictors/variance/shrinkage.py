"""Linear shrinkage covariance estimator with a pluggable target."""

from enum import IntEnum

import numpy as np

from ._base import VariancePredictor
from ._common import (
    correlation_from_covariance,
    schafer_strimmer_alpha,
    sample_covariance,
    single_index_covariance,
)


class Target(IntEnum):
    r"""Shrinkage target selector.

    The three targets surveyed in Pantaleo et al. (2010), Section III.D:

    - ``COMMON_COVARIANCE`` - diagonal = average sample variance,
      off-diagonal = average sample covariance.
    - ``CONSTANT_CORRELATION`` - diagonal = sample variances, off-diagonal
      = average off-diagonal sample correlation times the std outer product.
    - ``SINGLE_INDEX`` - single-index factor-model covariance using the
      equal-weighted cross-sectional mean as the market proxy.
    """

    COMMON_COVARIANCE = 1
    CONSTANT_CORRELATION = 2
    SINGLE_INDEX = 3


class Shrinkage(VariancePredictor[np.ndarray]):
    r"""Linear-shrinkage covariance estimator with a pluggable target.

    Computes \(\Sigma = \alpha F + (1 - \alpha) S\) where \(S\) is the
    sample covariance and \(F\) is one of the three structured targets.
    The intensity \(\alpha\) is estimated analytically via the
    Schäfer-Strimmer (2005) element-wise unbiased estimator.  Ignores
    features.
    """

    def __init__(self, *, target: Target = Target.COMMON_COVARIANCE, verbose: bool = False, **kwargs) -> None:
        target = Target(target)
        super().__init__(
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


def build(**kwargs) -> Shrinkage:
    """Construct a :class:`Shrinkage` covariance predictor.

    Build kwargs
    ------------
    num_stocks : int
    num_features : int
    universe_size : int
    target_offset : int
    target : int | Target, optional (default 1 == COMMON_COVARIANCE)
        Shrinkage target: 1=COMMON_COVARIANCE, 2=CONSTANT_CORRELATION,
        3=SINGLE_INDEX.
    verbose : bool, optional (default False)
    refit_every : int, optional (default 1)
    max_periods : int | None, optional
    min_periods : int | None, optional
    """
    return Shrinkage(**kwargs)
