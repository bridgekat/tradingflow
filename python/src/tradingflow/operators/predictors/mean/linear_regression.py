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
        features,
        adjusted_prices,
        *,
        rebalance_period: int,
        max_samples: int,
        verbose: bool = False,
    ) -> None:
        self._verbose = verbose
        super().__init__(
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

    Returns coefficient vector ``(num_features + 1,)`` (intercept last),
    or a zero vector if the design matrix is rank-deficient.
    """
    m, n = x.shape
    x = np.column_stack([x, np.ones(m)])

    try:
        q, r = np.linalg.qr(x, mode="reduced")

        if np.any(np.abs(np.diag(r)) < 1e-12):
            if verbose:
                print(f"  regression: design matrix is rank-deficient (diag R={np.diag(r)})")
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

    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"  regression: linear algebra error encountered ({e})")
        return np.zeros(n + 1)


def _predict_fn(state: MeanPredictorState[np.ndarray], x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict returns for all stocks using the linear model."""
    y = x @ params[:-1] + params[-1]
    y = np.clip(y, -20.0, 20.0)  # Clamp to prevent overflow in exp.
    return np.exp(y) - 1.0
