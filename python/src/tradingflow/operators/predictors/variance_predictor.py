"""Abstract variance (covariance matrix) predictor operator."""

from typing import Callable
from dataclasses import dataclass, field

import numpy as np

from ...views import ArrayView
from ...operator import Operator, Notify
from ...types import Array, Handle, NodeKind


@dataclass(slots=True)
class VariancePredictorState[T]:
    num_stocks: int
    num_features: int
    rebalance_period: int
    max_samples: int
    fit_fn: Callable[[np.ndarray, np.ndarray], T]
    predict_fn: Callable[["VariancePredictorState[T]", np.ndarray, T], np.ndarray]

    tick_count: int = 0
    features_list: list[np.ndarray] = field(default_factory=list)
    prices_list: list[np.ndarray] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


class VariancePredictor[T](
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Array[np.float64]], Handle[Array[np.float64]]],
        Handle[Array[np.float64]],
        VariancePredictorState[T],
    ]
):
    """Abstract covariance matrix predictor.

    Runs on every tick.  Accumulates per-tick cross-sectional price and
    feature snapshots, and every ``rebalance_period`` ticks builds a
    return matrix, calls ``fit_fn`` and ``predict_fn``, and outputs the
    predicted covariance matrix.

    Unlike [`MeanPredictor`][tradingflow.operators.predictors.MeanPredictor]
    which pools individual ``(time, stock)`` samples, this operator builds
    **cross-sectional return vectors** — each row of the return matrix is
    all stocks' returns at one time point.

    Parameters
    ----------
    universe
        Universe weights, shape ``(num_stocks,)``.
    features
        Stacked features, shape ``(num_stocks, num_features)``.
    adjusted_prices
        Stacked forward-adjusted close prices, shape ``(num_stocks,)``.
    fit_fn
        ``(x, y) -> params``.  Feature array ``x`` of shape
        ``(T, N, F)`` and return matrix ``y`` of shape ``(T, N)``.
    predict_fn
        ``(state, features, params) -> covariances``.  Current features
        of shape ``(N, F)`` and fitted params.
    rebalance_period
        Produce output every N ticks.
    """

    def __init__(
        self,
        universe: Handle,
        features: Handle,
        adjusted_prices: Handle,
        *,
        fit_fn: Callable[[np.ndarray, np.ndarray], T],
        predict_fn: Callable[[VariancePredictorState[T], np.ndarray, T], np.ndarray],
        rebalance_period: int,
        max_samples: int = 1000,
    ) -> None:
        assert len(universe.shape) == 1
        assert len(features.shape) == 2
        assert len(adjusted_prices.shape) == 1
        assert universe.shape[0] == features.shape[0] == adjusted_prices.shape[0]

        self._num_stocks = features.shape[0]
        self._num_features = features.shape[1]
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._rebalance_period = rebalance_period
        self._max_samples = max_samples

        super().__init__(
            inputs=(universe, features, adjusted_prices),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_stocks, self._num_stocks),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> VariancePredictorState[T]:
        return VariancePredictorState(
            num_stocks=self._num_stocks,
            num_features=self._num_features,
            rebalance_period=self._rebalance_period,
            max_samples=self._max_samples,
            fit_fn=self._fit_fn,
            predict_fn=self._predict_fn,
        )

    @staticmethod
    def compute(
        state: VariancePredictorState[T],
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        notify: Notify,
    ) -> bool:
        N = state.num_stocks
        F = state.num_features
        K = state.rebalance_period

        universe = inputs[0].value()
        features = inputs[1].value()
        prices = inputs[2].value()

        # Append snapshot (store array reference, no copy).
        state.features_list.append(features)
        state.prices_list.append(prices)

        n_ticks = len(state.prices_list)

        # Only produce output every rebalance_period ticks.
        state.tick_count += 1
        if state.tick_count < state.rebalance_period:
            return False
        state.tick_count = 0

        # Need at least rebalance_period + 1 ticks to have valid returns.
        if n_ticks <= K:
            return False

        # Filter to stocks currently in the universe with valid features.
        mask = (universe > 0) & np.isfinite(features).all(axis=1)
        n_univ = int(mask.sum())

        # Build cross-sectional return matrix and feature array for
        # universe stocks, subsampling time rows if needed.
        n_periods = n_ticks - K
        if n_periods > state.max_samples:
            ts = state.rng.choice(n_periods, state.max_samples, replace=False)
            ts.sort()
        else:
            ts = np.arange(n_periods)

        m = len(ts)
        x = np.empty((m, n_univ, F), dtype=np.float64)
        y = np.empty((m, n_univ), dtype=np.float64)
        for i, t in enumerate(ts):
            x[i] = state.features_list[t][mask]
            p0 = state.prices_list[t][mask]
            p1 = state.prices_list[t + K][mask]
            y[i] = p1 / np.maximum(p0, 1e-10) - 1.0

        # Fit and predict (symmetric with MeanPredictor: x first, y second).
        params = state.fit_fn(x, y)
        sigma_sub = state.predict_fn(state, features[mask], params)

        # Write back into full (N, N) matrix.
        sigma = np.zeros((N, N), dtype=np.float64)
        sigma[np.ix_(mask, mask)] = sigma_sub

        output.write(sigma)
        return True
