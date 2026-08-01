from enum import IntEnum

import numpy as np

from ._base import VariancePredictor, covariance_predictor
from ._common import (
    correlation_from_covariance,
    sample_covariance,
    schafer_strimmer_alpha,
    single_index_covariance,
)


class Target(IntEnum):
    r"""Shrinkage target selector — the three surveyed in Pantaleo et al.
    (2010), Section III.D.

    - `COMMON_COVARIANCE`: diagonal is the average sample variance,
      off-diagonal the average sample covariance.
    - `CONSTANT_CORRELATION`: diagonal is the sample variances, off-diagonal
      the average off-diagonal correlation times the outer product of the
      standard deviations.
    - `SINGLE_INDEX`: the single-index factor-model covariance.
    """

    COMMON_COVARIANCE = 1
    CONSTANT_CORRELATION = 2
    SINGLE_INDEX = 3


def common_covariance_target(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    n = s.shape[0]
    off = ~np.eye(n, dtype=bool)
    avg_var = float(np.mean(np.diag(s)))
    target = np.full((n, n), float(s[off].mean()) if n > 1 else avg_var)
    np.fill_diagonal(target, avg_var)
    return target


def constant_correlation_target(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    n = s.shape[0]
    corr, stds = correlation_from_covariance(s)
    off = ~np.eye(n, dtype=bool)
    r_bar = float(corr[off].mean()) if n > 1 else 1.0
    target = r_bar * np.outer(stds, stds)
    np.fill_diagonal(target, np.diag(s))
    return target


TARGETS = {
    Target.COMMON_COVARIANCE: common_covariance_target,
    Target.CONSTANT_CORRELATION: constant_correlation_target,
    Target.SINGLE_INDEX: lambda y, s: single_index_covariance(y),
}


def fit(y: np.ndarray, *, target: Target) -> np.ndarray:
    r"""Linear shrinkage \(\Sigma = \alpha F + (1 - \alpha) S\) of the sample
    covariance \(S\) toward a structured target \(F\), with the intensity
    \(\alpha\) estimated analytically by the Schäfer-Strimmer (2005)
    element-wise unbiased estimator."""
    s, centered, finite = sample_covariance(y)
    f = TARGETS[target](y, s)
    alpha, _ = schafer_strimmer_alpha(s, f, centered, finite)
    return alpha * f + (1.0 - alpha) * s


def build(*, target: int | Target = Target.COMMON_COVARIANCE, **kwargs) -> VariancePredictor:
    """Constructs a linear-shrinkage covariance predictor."""
    target = Target(target)
    return covariance_predictor(lambda y: fit(y, target=target), **kwargs)
