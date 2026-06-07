"""Ordinary least squares (OLS) pooled linear regression mean predictor."""

from dataclasses import dataclass

import numpy as np

from ._base import MeanPredictor, MeanPredictorState, pool_and_subsample


@dataclass(slots=True)
class LinearRegressionParams:
    """Fitted OLS coefficients with the pooled standardization scaler."""

    beta: np.ndarray
    intercept: float
    x_mean: np.ndarray
    x_std: np.ndarray


class LinearRegression(MeanPredictor[LinearRegressionParams]):
    r"""Pooled OLS mean predictor on pool-standardized features and target.

    Solves \(\min_{\beta} \|y - \tilde{X}\beta - \bar{y}\|_2^2\) via QR
    decomposition on the standardized design matrix.  Features and target
    are pool-standardized over the training window; for unpenalized OLS
    the results are mathematically identical to fitting on raw features
    but more numerically stable.
    """

    def __init__(
        self, *, verbose: bool = False, max_samples: int | None = None, subsample_seed: int = 0, **kwargs
    ) -> None:
        self._verbose = verbose
        super().__init__(
            fit_fn=lambda x, y: _fit_fn(
                x, y, verbose=verbose, max_samples=max_samples, seed=subsample_seed
            ),
            predict_fn=_predict_fn,
            **kwargs,
        )


def _fit_fn(
    x: np.ndarray, y: np.ndarray, *, verbose: bool, max_samples: int | None = None, seed: int = 0
) -> LinearRegressionParams:
    """Fit OLS on a pooled, standardized design matrix via QR."""
    F = x.shape[2]
    x, y = pool_and_subsample(x, y, max_samples, seed)
    m = len(y)

    if m == 0:
        if verbose:
            print("  linear_regression: no valid samples after NaN filter")
        return LinearRegressionParams(np.zeros(F), 0.0, np.zeros(F), np.ones(F))

    if verbose:
        print(f"  linear_regression: x has shape {x.shape} and range [{x.min():.4f}, {x.max():.4f}]")
        print(f"  linear_regression: y has shape {y.shape} and range [{y.min():.4f}, {y.max():.4f}]")

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

    fallback = LinearRegressionParams(np.zeros(F), y_mean, x_mean, x_std)

    # Standardized OLS: solve Z beta_tilde = y_normalized via QR.
    q, r = np.linalg.qr(x_normalized, mode="reduced")

    if q.shape[1] < F:
        print(
            f"  linear_regression: design matrix is rank-deficient "
            f"(rank={q.shape[1]}, expected={F}), using zero coefficients"
        )
        return fallback

    try:
        beta_tilde = np.linalg.solve(r, q.T @ y_normalized)
    except np.linalg.LinAlgError as e:
        print(f"  linear_regression: QR back-substitution failed ({e}), using zero coefficients")
        return fallback

    if not np.all(np.isfinite(beta_tilde)):
        print(f"  linear_regression: non-finite coefficients (beta={beta_tilde}), using zero coefficients")
        return fallback

    # Recover coefficients in the original-y scale.
    beta = y_std * beta_tilde

    if verbose:
        rss = float(np.sum((y - y_mean - x_normalized @ beta) ** 2))
        tss = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        n_nonzero = int(np.sum(np.abs(beta) > 1e-8))
        print(f"  linear_regression: {m} samples, R2={r2:.4f}, {n_nonzero}/{F} nonzero beta")

    return LinearRegressionParams(beta=beta, intercept=y_mean, x_mean=x_mean, x_std=x_std)


def _predict_fn(
    state: MeanPredictorState[LinearRegressionParams],
    x: np.ndarray,
    params: LinearRegressionParams,
) -> np.ndarray:
    """Apply the fitted standardization scaler, then the linear model."""
    z = (x - params.x_mean) / params.x_std
    return z @ params.beta + params.intercept


def build(**kwargs) -> LinearRegression:
    """Construct a :class:`LinearRegression` mean predictor.

    Build kwargs
    ------------
    num_stocks : int
    num_features : int
    universe_size : int
    target_offset : int
    verbose : bool, optional (default False)
    refit_every : int, optional (default 1)
    max_periods : int | None, optional
    min_periods : int | None, optional
    """
    return LinearRegression(**kwargs)
