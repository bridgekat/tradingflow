"""Linear regression mean predictor."""

import numpy as np

from ..mean_predictor import MeanPredictor, MeanPredictorState


class LinearRegression(MeanPredictor[np.ndarray]):
    """Mean predictor using pooled OLS regression.

    Fits ``y = X @ beta + intercept`` via QR decomposition on each
    rebalance tick, where ``y`` is the forward log-return over
    ``predict_horizon`` days and ``X`` is the feature matrix.

    Parameters
    ----------
    features
        Stacked features, shape ``(num_stocks, num_features)``.
    adjusted_prices
        Stacked forward-adjusted close prices, shape ``(num_stocks,)``.
    rebalance_period
        Produce output every N ticks.
    max_samples
        Maximum regression samples.
    verbose
        If ``True``, print regression diagnostics to stdout.
    """

    def __init__(
        self,
        universe,
        features,
        adjusted_prices,
        *,
        rebalance_period: int,
        max_samples: int,
        verbose: bool = False,
    ) -> None:
        self._verbose = verbose
        super().__init__(
            universe,
            features,
            adjusted_prices,
            fit_fn=lambda x, y: _fit_fn(x, y, verbose=verbose),
            predict_fn=_predict_fn,
            rebalance_period=rebalance_period,
            max_samples=max_samples,
        )

    def init(self, inputs: tuple, timestamp: int) -> MeanPredictorState:
        state = super().init(inputs, timestamp)
        return state


def _fit_fn(x: np.ndarray, y: np.ndarray, *, verbose: bool = False) -> np.ndarray:
    """Fit OLS via QR decomposition.

    Parameters
    ----------
    x
        Feature tensor of shape ``(T, N, F)``.
    y
        Return matrix of shape ``(T, N)``.

    Returns coefficient vector ``(num_features + 1,)`` (intercept last),
    or a zero vector if the design matrix is rank-deficient.
    """

    # Flatten cross-sectional structure for pooled regression.
    T, N, F = x.shape
    x = x.reshape(T * N, F)
    y = y.reshape(T * N)

    # Drop non-finite samples.
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[valid], y[valid]

    m = len(y)
    n = F
    x = np.column_stack([x, np.ones(m)])
    q, r = np.linalg.qr(x, mode="reduced")

    if q.shape[1] < n + 1:
        if verbose:
            print(f"  regression: design matrix is rank-deficient (rank={q.shape[1]}, expected={n + 1})")
        return np.zeros(n + 1)

    params = np.linalg.solve(r, q.T @ y)

    if not np.all(np.isfinite(params)):
        if verbose:
            print(f"  regression: non-finite parameters encountered (params={params})")
        return np.zeros(n + 1)

    if verbose:
        rss = np.sum((y - x @ params) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        print(f"  regression: {m} samples, R2={r2:.4f}")
    return params


def _predict_fn(state: MeanPredictorState[np.ndarray], x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict returns for all stocks using the linear model."""
    return x @ params[:-1] + params[-1]
