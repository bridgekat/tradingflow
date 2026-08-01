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
    r"""Fits Lasso on a pooled, standardized design matrix via CVXPY (SCS).

    Solves \(\min_\beta \frac{1}{m}\|y - \tilde{X}\beta - \bar{y}\|_2^2 +
    \alpha\|\beta\|_1\). Standardization, sample-size invariance and
    target-scale invariance all match `ridge`; only the penalty and hence the
    solver differ, since L1 has no closed form to reduce to a QR.

    The import is deferred because CVXPY is not available on every interpreter
    this package runs on — the module stays importable, and only an actual fit
    raises.
    """
    import cvxpy as cp

    f = x.shape[2]
    x, y = pool_and_subsample(x, y, max_samples, seed)
    m = len(y)
    if m == 0:
        return LinearParams(np.zeros(f), 0.0, np.zeros(f), np.ones(f))

    z, target, x_mean, x_std, y_mean, y_std = standardize(x, y)
    fallback = LinearParams(np.zeros(f), y_mean, x_mean, x_std)

    beta = cp.Variable(f)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(z @ beta - target) / m + alpha * cp.norm1(beta))
    )
    try:
        problem.solve(solver=cp.SCS)
    except cp.SolverError:
        return fallback
    if beta.value is None:
        return fallback

    solved = np.asarray(beta.value, dtype=np.float64)
    if not np.all(np.isfinite(solved)):
        return fallback

    return LinearParams(y_std * solved, y_mean, x_mean, x_std)


def build(
    *,
    alpha: float = 1.0,
    max_samples: int | None = None,
    subsample_seed: int = 0,
    **kwargs,
) -> MeanPredictor:
    """Constructs a pooled Lasso mean predictor. Requires CVXPY."""
    assert alpha >= 0.0, "Lasso alpha must be non-negative"
    return MeanPredictor(
        fit=lambda x, y: fit(x, y, alpha=float(alpha), max_samples=max_samples, seed=subsample_seed),
        predict=linear_predict,
        **kwargs,
    )
