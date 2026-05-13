"""Ordinary least squares (OLS) pooled linear regression mean predictor."""

from dataclasses import dataclass

import numpy as np

from ..mean_predictor import MeanPredictor, MeanPredictorState


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
    decomposition on the standardized design matrix \(\tilde{X}\).
    Predictions and \(R^2\) are identical to OLS on raw features:
    standardization is a pure affine reparameterization in the
    unpenalized case.  The reason for standardizing anyway is
    structural symmetry with
    [`Ridge`][tradingflow.operators.predictors.mean.ridge.Ridge] and
    [`Lasso`][tradingflow.operators.predictors.mean.lasso.Lasso], plus
    the numerical robustness it gives on poorly-scaled inputs (the
    condition number is not squared).

    ## Standardization

    Features and target are both pool-standardized over the training
    window (pooled mean and population standard deviation).  Constant
    columns / constant target use a `std = 1` fallback so they
    contribute nothing to the fit; the corresponding `beta_j` stays at
    zero and predictions become constant at `y_mean`.  For linear
    regression without regularization, the results are mathematically
    identical to fitting on raw features, but can be more numerically
    stable.

    Cross-sectional
    [`Standardize`][tradingflow.operators.num.standardize.Standardize]
    upstream is complementary, not a substitute: it reshapes each row's
    distribution but does not give a time-stable per-feature scale.

    Parameters
    ----------
    universe
        Universe weights, shape `(num_stocks,)`.
    features_series
        Recorded features series, element shape `(num_stocks, num_features)`.
    target_series
        Recorded target series, element shape `(num_stocks,)`.
    verbose
        If `True`, print regression diagnostics to stdout.
    **kwargs
        Forwarded to [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor].
    """

    def __init__(
        self,
        universe,
        features_series,
        target_series,
        *,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._verbose = verbose
        super().__init__(
            universe,
            features_series,
            target_series,
            fit_fn=lambda x, y: _fit_fn(x, y, verbose=verbose),
            predict_fn=_predict_fn,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray, *, verbose: bool) -> LinearRegressionParams:
    """Fit OLS on a pooled, standardized design matrix via QR."""
    M, N, F = x.shape
    x = x.reshape(M * N, F)
    y = y.reshape(M * N)

    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[valid], y[valid]
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
