import numpy as np

from ._base import LinearParams, MeanPredictor, linear_predict, pool_and_subsample, standardize


def fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    max_samples: int | None = None,
    seed: int = 0,
) -> LinearParams:
    r"""Fits Ridge on a pooled, standardized design matrix via augmented QR.

    Solves \(\min_\beta \frac{1}{m}\|y - \tilde{X}\beta - \bar{y}\|_2^2 +
    \alpha\|\beta\|_2^2\) by reducing to OLS on the augmented design
    \([\tilde{X}; \sqrt{\alpha m} I]\) against \([y - \bar{y}; 0]\). The
    \(1/m\) prefactor together with pool-standardization is what makes `alpha`
    a dimensionless dial: it means the same thing at any sample size and any
    target scale.
    """
    f = x.shape[2]
    x, y = pool_and_subsample(x, y, max_samples, seed)
    m = len(y)
    if m == 0:
        return LinearParams(np.zeros(f), 0.0, np.zeros(f), np.ones(f))

    z, target, x_mean, x_std, y_mean, y_std = standardize(x, y)
    fallback = LinearParams(np.zeros(f), y_mean, x_mean, x_std)

    q, r = np.linalg.qr(
        np.vstack([z, np.sqrt(alpha * m) * np.eye(f)]),
        mode="reduced",
    )
    try:
        beta = np.linalg.solve(r, q.T @ np.concatenate([target, np.zeros(f)]))
    except np.linalg.LinAlgError:
        return fallback
    if not np.all(np.isfinite(beta)):
        return fallback

    return LinearParams(y_std * beta, y_mean, x_mean, x_std)


def build(
    *,
    alpha: float = 1.0,
    max_samples: int | None = None,
    subsample_seed: int = 0,
    **kwargs,
) -> MeanPredictor:
    """Constructs a pooled Ridge mean predictor."""
    assert alpha >= 0.0, "Ridge alpha must be non-negative"
    return MeanPredictor(
        fit=lambda x, y: fit(x, y, alpha=float(alpha), max_samples=max_samples, seed=subsample_seed),
        predict=linear_predict,
        **kwargs,
    )
