"""Linear regression mean predictor."""

import numpy as np

from ..mean_predictor import MeanPredictor, MeanPredictorState


class LinearRegression(MeanPredictor[np.ndarray]):
    r"""Mean predictor using pooled OLS regression.

    Fits \(y = X \beta + \mathrm{intercept}\) via QR decomposition on
    each rebalance tick, where \(y\) is the aligned target row and
    \(X\) is the feature matrix.  The meaning of the prediction is
    whatever the upstream target series represents (linear returns,
    log returns, a custom signal, etc.).

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

    def init(self, inputs: tuple, timestamp: int) -> MeanPredictorState:
        state = super().init(inputs, timestamp)
        return state


def _fit_fn(x: np.ndarray, y: np.ndarray, *, verbose: bool = False) -> np.ndarray:
    """Fit OLS via QR decomposition.

    Parameters
    ----------
    x
        Feature tensor of shape `(M, N, F)`.
    y
        Return matrix of shape `(M, N)`.

    Returns
    -------
    np.ndarray
        Coefficient vector `(num_features + 1,)` (intercept last),
        or a zero vector if the design matrix is rank-deficient.
    """

    # Flatten cross-sectional structure for pooled regression.
    M, N, F = x.shape
    x = x.reshape(M * N, F)
    y = y.reshape(M * N)

    # Drop non-finite samples.
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if verbose:
        print(f"  regression: x has shape {x.shape} and range [{x.min():.4f}, {x.max():.4f}]")
        print(f"  regression: y has shape {y.shape} and range [{y.min():.4f}, {y.max():.4f}]")

    m = len(y)
    x = np.column_stack([x, np.ones(m)])
    q, r = np.linalg.qr(x, mode="reduced")

    if q.shape[1] < F + 1:
        print(f"  regression: design matrix is rank-deficient (rank={q.shape[1]}, expected={F + 1})")
        return np.zeros(F + 1)

    params = np.linalg.solve(r, q.T @ y)

    if not np.all(np.isfinite(params)):
        print(f"  regression: non-finite parameters encountered (params={params})")
        return np.zeros(F + 1)

    if verbose:
        rss = np.sum((y - x @ params) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        print(f"  regression: {m} samples, R2={r2:.4f}")
    return params


def _predict_fn(state: MeanPredictorState[np.ndarray], x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict returns for all stocks using the linear model."""
    return x @ params[:-1] + params[-1]
