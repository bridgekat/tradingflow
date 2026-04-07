"""Abstract mean-return predictor operator."""

from typing import Callable
from dataclasses import dataclass, field

import numpy as np

from ...views import ArrayView
from ...operator import Operator, Notify
from ...types import Array, Handle, NodeKind


@dataclass(slots=True)
class MeanPredictorState[T]:
    num_stocks: int
    num_features: int
    rebalance_period: int
    max_samples: int
    fit_fn: Callable[[np.ndarray, np.ndarray], T]
    predict_fn: Callable[["MeanPredictorState[T]", np.ndarray, T], np.ndarray]

    tick_count: int = 0
    features_list: list[np.ndarray] = field(default_factory=list)
    prices_list: list[np.ndarray] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


class MeanPredictor[T](
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Array[np.float64]], Handle[Array[np.float64]]],
        Handle[Array[np.float64]],
        MeanPredictorState[T],
    ]
):
    """Abstract mean-return predictor.

    Runs on every tick.  Accumulates per-tick cross-sectional feature
    and price snapshots, and every ``rebalance_period`` ticks builds
    a return matrix and feature array, subsamples time rows, calls
    ``fit_fn`` and ``predict_fn``, and outputs predicted returns.

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
        ``(state, features, params) -> returns``.  Current features
        of shape ``(N, F)`` and fitted params.
    rebalance_period
        Produce output every N ticks.
    max_samples
        Maximum number of time rows to feed to ``fit_fn``.
    """

    def __init__(
        self,
        universe: Handle,
        features: Handle,
        adjusted_prices: Handle,
        *,
        fit_fn: Callable[[np.ndarray, np.ndarray], T],
        predict_fn: Callable[[MeanPredictorState[T], np.ndarray, T], np.ndarray],
        rebalance_period: int,
        max_samples: int,
    ) -> None:
        assert len(universe.shape) == 1
        assert len(features.shape) == 2
        assert len(adjusted_prices.shape) == 1
        assert universe.shape[0] == features.shape[0] == adjusted_prices.shape[0]

        self._num_stocks = features.shape[0]
        self._num_features = features.shape[-1]
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._rebalance_period = rebalance_period
        self._max_samples = max_samples

        super().__init__(
            inputs=(universe, features, adjusted_prices),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_stocks,),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> MeanPredictorState[T]:
        return MeanPredictorState(
            num_stocks=self._num_stocks,
            num_features=self._num_features,
            fit_fn=self._fit_fn,
            predict_fn=self._predict_fn,
            rebalance_period=self._rebalance_period,
            max_samples=self._max_samples,
        )

    @staticmethod
    def compute(
        state: MeanPredictorState[T],
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
            p0[~(p0 > 0)] = np.nan
            p1[~(p1 > 0)] = np.nan
            y[i] = p1 / p0 - 1.0

        # Fit and predict only for universe stocks with valid features.
        params = state.fit_fn(x, y)
        mu_sub = state.predict_fn(state, features[mask], params)

        # Write back into full (N,) array.
        mu = np.zeros(N, dtype=np.float64)
        mu[mask] = mu_sub

        output.write(mu)
        return True
