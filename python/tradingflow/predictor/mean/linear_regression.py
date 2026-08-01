import numpy as np

from ._base import LinearParams, MeanPredictor, linear_predict, pool_and_subsample, standardize


def fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int | None = None,
    seed: int = 0,
) -> LinearParams:
    r"""Fits OLS on a pooled, standardized design matrix via QR.

    Solves \(\min_\beta \|y - \tilde{X}\beta - \bar{y}\|_2^2\). Standardizing
    is mathematically neutral for unpenalized OLS but numerically kinder, and
    it keeps the scaler contract identical to the penalized predictors.

    A rank-deficient design has no unique solution, so rather than take the
    minimum-norm one the fit falls back to zero coefficients — constant
    predictions at the target mean.
    """
    f = x.shape[2]
    x, y = pool_and_subsample(x, y, max_samples, seed)
    if len(y) == 0:
        return LinearParams(np.zeros(f), 0.0, np.zeros(f), np.ones(f))

    z, target, x_mean, x_std, y_mean, y_std = standardize(x, y)
    fallback = LinearParams(np.zeros(f), y_mean, x_mean, x_std)

    q, r = np.linalg.qr(z, mode="reduced")
    if q.shape[1] < f:
        return fallback
    try:
        beta = np.linalg.solve(r, q.T @ target)
    except np.linalg.LinAlgError:
        return fallback
    if not np.all(np.isfinite(beta)):
        return fallback

    return LinearParams(y_std * beta, y_mean, x_mean, x_std)


def build(*, max_samples: int | None = None, subsample_seed: int = 0, **kwargs) -> MeanPredictor:
    """Constructs a pooled OLS mean predictor."""
    return MeanPredictor(
        fit=lambda x, y: fit(x, y, max_samples=max_samples, seed=subsample_seed),
        predict=linear_predict,
        **kwargs,
    )
