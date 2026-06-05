"""LASSO (L1-penalized) pooled linear regression mean predictor.

BLOCKER: the L1 fit uses CVXPY (SCS backend), which is not installed on the
free-threaded interpreter (and is not currently free-threaded-safe).  L1 has
no closed form, so the QR reduction used by Ridge does not apply.  Per the
porting rules we do NOT add a numpy-only reimplementation.  The module is
kept importable on the ft host (the ``import cvxpy`` is deferred into
``_fit_fn``); attempting an actual rebalance/fit will raise ``ImportError``
until cvxpy is available.  ``predict_fn`` and the standardization scaler are
pure numpy and fully portable.
"""

from dataclasses import dataclass

import numpy as np

from ._base import MeanPredictor, MeanPredictorState


@dataclass(slots=True)
class LassoParams:
    """Fitted Lasso coefficients with the pooled standardization scaler."""

    beta: np.ndarray
    intercept: float
    x_mean: np.ndarray
    x_std: np.ndarray


class Lasso(MeanPredictor[LassoParams]):
    r"""Pooled L1-penalized regression mean predictor.

    Solves

    \[
        \min_{\beta} \;\;
        \tfrac{1}{m}\,\|y - \tilde{X}\beta - \bar{y}\|_2^2 + \alpha \|\beta\|_1
    \]

    via CVXPY (SCS backend).  Standardization, sample-size invariance, and
    target-rescale invariance match Ridge; only the solver differs.  On
    solver failure the fit falls back to zero coefficients.
    """

    def __init__(self, *, alpha: float = 1.0, verbose: bool = False, **kwargs) -> None:
        assert alpha >= 0.0, "Lasso alpha must be non-negative"
        self._alpha = float(alpha)
        self._verbose = verbose
        super().__init__(
            fit_fn=lambda x, y: _fit_fn(x, y, alpha=self._alpha, verbose=verbose),
            predict_fn=_predict_fn,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray, *, alpha: float, verbose: bool) -> LassoParams:
    """Fit Lasso on a pooled, standardized design matrix via CVXPY/SCS."""
    # Deferred import: cvxpy is unavailable on the free-threaded interpreter.
    import cvxpy as cp

    M, N, F = x.shape
    x = x.reshape(M * N, F)
    y = y.reshape(M * N)

    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[valid], y[valid]
    m = len(y)

    if m == 0:
        if verbose:
            print("  lasso: no valid samples after NaN filter")
        return LassoParams(np.zeros(F), 0.0, np.zeros(F), np.ones(F))

    if verbose:
        print(f"  lasso: x has shape {x.shape} and range [{x.min():.4f}, {x.max():.4f}]")
        print(f"  lasso: y has shape {y.shape} and range [{y.min():.4f}, {y.max():.4f}]")

    # Pool-standardize features and target symmetrically: pooled mean,
    # population std (ddof=0), with constant-column / constant-target
    # fallback std=1 so the corresponding standardized values are zero
    # and contribute nothing to the solve (the operator then produces
    # zero coefficients and constant predictions at y_mean).
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0)
    x_std = np.where(x_std > 0, x_std, 1.0)
    x_normalized = (x - x_mean) / x_std

    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    y_std = y_std if y_std > 0 else 1.0
    y_normalized = (y - y_mean) / y_std

    fallback = LassoParams(np.zeros(F), y_mean, x_mean, x_std)

    # Sample-size-invariant Lasso: (1/m) ||y_normalized - Z beta_tilde||^2 + alpha ||beta_tilde||_1.
    beta_var = cp.Variable(F)
    objective = cp.Minimize(cp.sum_squares(x_normalized @ beta_var - y_normalized) / m + alpha * cp.norm1(beta_var))
    prob = cp.Problem(objective)
    try:
        prob.solve(solver=cp.SCS)
    except cp.SolverError as e:
        print(f"  lasso: solver failed ({e}), using zero coefficients")
        return fallback

    if beta_var.value is None:
        print(f"  lasso: no solution (status={prob.status}), using zero coefficients")
        return fallback

    beta_tilde = np.asarray(beta_var.value, dtype=np.float64)
    if not np.all(np.isfinite(beta_tilde)):
        print(f"  lasso: non-finite coefficients (beta={beta_tilde}), using zero coefficients")
        return fallback

    # Recover coefficients in the original-y scale.
    beta = y_std * beta_tilde

    if verbose:
        rss = float(np.sum((y - y_mean - x_normalized @ beta) ** 2))
        tss = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        n_nonzero = int(np.sum(np.abs(beta) > 1e-8))
        print(
            f"  lasso: {m} samples, alpha={alpha:.4g}, R2={r2:.4f}, "
            f"{n_nonzero}/{F} nonzero beta, status={prob.status}"
        )

    return LassoParams(beta=beta, intercept=y_mean, x_mean=x_mean, x_std=x_std)


def _predict_fn(state: MeanPredictorState[LassoParams], x: np.ndarray, params: LassoParams) -> np.ndarray:
    """Apply the fitted standardization scaler, then the linear model."""
    z = (x - params.x_mean) / params.x_std
    return z @ params.beta + params.intercept


def build(**kwargs) -> Lasso:
    """Construct a :class:`Lasso` mean predictor.

    BLOCKER: requires cvxpy (unavailable on the ft interpreter).  The op
    constructs and imports fine; the first fit raises ImportError.

    Build kwargs
    ------------
    num_stocks : int
    num_features : int
    universe_size : int
    target_offset : int
    alpha : float, optional (default 1.0)
        Non-negative L1 penalty strength (dimensionless).
    verbose : bool, optional (default False)
    refit_every : int, optional (default 1)
    max_periods : int | None, optional
    min_periods : int | None, optional
    """
    return Lasso(**kwargs)
