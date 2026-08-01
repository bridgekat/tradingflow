r"""Random matrix theory (RMT) covariance predictors.

Both estimators diagonalize the sample correlation matrix and suppress the
eigenvalues below the Laloux-corrected Marchenko-Pastur upper bound

\[
\lambda_\max = \sigma^2 \left(1 + N/T + 2\sqrt{N/T}\right),
\quad \sigma^2 = 1 - \lambda_1 / N,
\]

which is where the eigenvalue spectrum of pure noise ends — everything below it
is indistinguishable from a random correlation and carries no signal worth
keeping. They differ in what replaces those eigenvalues: `zero` discards them
(Rosenow et al., 2002), `mean` replaces them with their mean and so preserves
the trace (Potters et al., 2005).
"""

import numpy as np

from ._base import VariancePredictor, covariance_predictor
from ._common import correlation_from_covariance, sample_covariance


def fit(y: np.ndarray, *, mode: str) -> np.ndarray:
    """RMT-filters the sample correlation and rescales to a covariance."""
    t, n = y.shape
    s, _, _ = sample_covariance(y)
    corr, stds = correlation_from_covariance(s)

    if t <= 0:
        return s

    # Symmetrize for numerical safety before the eigendecomposition.
    corr = (corr + corr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(corr)  # ascending

    q = n / t
    sigma2 = max(1.0 - float(eigvals[-1]) / n, 0.0)
    below = eigvals < sigma2 * (1.0 + q + 2.0 * np.sqrt(q))

    filtered = eigvals.copy()
    if mode == "zero":
        filtered[below] = 0.0
    elif mode == "mean":
        if below.any():
            filtered[below] = eigvals[below].mean()
    else:
        raise ValueError(f"unknown RMT mode {mode!r}")

    h = (eigvecs * filtered) @ eigvecs.T

    if mode == "zero":
        result = h.copy()
    else:
        # Renormalize to a unit diagonal, preserving the off-diagonal
        # structure the mean replacement produced.
        scale = np.sqrt(np.maximum(np.diag(h), 1e-30))
        result = h / np.outer(scale, scale)
    np.fill_diagonal(result, 1.0)

    return result * np.outer(stds, stds)


def build(*, mode: str = "zero", **kwargs) -> VariancePredictor:
    """Constructs an RMT covariance predictor: `mode` is `"zero"` (RMT-0) or
    `"mean"` (RMT-M)."""
    assert mode in ("zero", "mean"), f"unknown RMT mode {mode!r}"
    return covariance_predictor(lambda y: fit(y, mode=mode), **kwargs)
