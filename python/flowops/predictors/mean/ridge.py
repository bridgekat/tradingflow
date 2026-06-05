"""Ridge (L2-penalized) pooled linear regression mean predictor."""

from dataclasses import dataclass

import numpy as np

from ._base import MeanPredictor, MeanPredictorState


@dataclass(slots=True)
class RidgeParams:
    """Fitted Ridge coefficients with the pooled standardization scaler."""

    beta: np.ndarray
    intercept: float
    x_mean: np.ndarray
    x_std: np.ndarray


class Ridge(MeanPredictor[RidgeParams]):
    r"""Pooled L2-penalized regression mean predictor.

    Solves

    \[
        \min_{\beta} \;\;
        \tfrac{1}{m}\,\|y - \tilde{X}\beta - \bar{y}\|_2^2 + \alpha \|\beta\|_2^2
    \]

    by reducing to OLS on the augmented matrix
    \([\tilde{X};\,\sqrt{\alpha m}\,I_F]\) with RHS
    \([y - \bar{y};\,0]\) and solving via QR.  The \(1/m\) prefactor plus
    pool-standardization make ``alpha`` a dimensionless dial.
    """

    def __init__(self, *, alpha: float = 1.0, verbose: bool = False, **kwargs) -> None:
        assert alpha >= 0.0, "Ridge alpha must be non-negative"
        self._alpha = float(alpha)
        self._verbose = verbose
        super().__init__(
            fit_fn=lambda x, y: _fit_fn(x, y, alpha=self._alpha, verbose=verbose),
            predict_fn=_predict_fn,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray, *, alpha: float, verbose: bool) -> RidgeParams:
    """Fit Ridge on a pooled, standardized design matrix via augmented QR."""
    M, N, F = x.shape
    x = x.reshape(M * N, F)
    y = y.reshape(M * N)

    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[valid], y[valid]
    m = len(y)

    if m == 0:
        if verbose:
            print("  ridge: no valid samples after NaN filter")
        return RidgeParams(np.zeros(F), 0.0, np.zeros(F), np.ones(F))

    if verbose:
        print(f"  ridge: x has shape {x.shape} and range [{x.min():.4f}, {x.max():.4f}]")
        print(f"  ridge: y has shape {y.shape} and range [{y.min():.4f}, {y.max():.4f}]")

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

    fallback = RidgeParams(np.zeros(F), y_mean, x_mean, x_std)

    # Sample-size-invariant Ridge: the loss is (1/m) ||y - Z beta||^2,
    # so the penalty contribution scales by m too.  Equivalent to OLS
    # on [Z; sqrt(alpha*m) I] beta_tilde = [y_normalized; 0].
    x_aug = np.vstack([x_normalized, np.sqrt(alpha * m) * np.eye(F)])
    y_aug = np.concatenate([y_normalized, np.zeros(F)])
    q, r = np.linalg.qr(x_aug, mode="reduced")

    try:
        beta_tilde = np.linalg.solve(r, q.T @ y_aug)
    except np.linalg.LinAlgError as e:
        print(f"  ridge: QR back-substitution failed ({e}), using zero coefficients")
        return fallback

    if not np.all(np.isfinite(beta_tilde)):
        print(f"  ridge: non-finite coefficients (beta={beta_tilde}), using zero coefficients")
        return fallback

    # Recover coefficients in the original-y scale.
    beta = y_std * beta_tilde

    if verbose:
        rss = float(np.sum((y - y_mean - x_normalized @ beta) ** 2))
        tss = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        n_nonzero = int(np.sum(np.abs(beta) > 1e-8))
        print(f"  ridge: {m} samples, alpha={alpha:.4g}, R2={r2:.4f}, {n_nonzero}/{F} nonzero beta")

    return RidgeParams(beta=beta, intercept=y_mean, x_mean=x_mean, x_std=x_std)


def _predict_fn(state: MeanPredictorState[RidgeParams], x: np.ndarray, params: RidgeParams) -> np.ndarray:
    """Apply the fitted standardization scaler, then the linear model."""
    z = (x - params.x_mean) / params.x_std
    return z @ params.beta + params.intercept


def build(**kwargs) -> Ridge:
    """Construct a :class:`Ridge` mean predictor.

    Build kwargs
    ------------
    num_stocks : int
    num_features : int
    universe_size : int
    target_offset : int
    alpha : float, optional (default 1.0)
        Non-negative L2 penalty strength (dimensionless).
    verbose : bool, optional (default False)
    refit_every : int, optional (default 1)
    max_periods : int | None, optional
    min_periods : int | None, optional
    """
    return Ridge(**kwargs)
