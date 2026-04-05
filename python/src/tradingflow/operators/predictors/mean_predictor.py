"""Abstract mean-return predictor operator."""

from typing import Callable
from dataclasses import dataclass, field

import numpy as np

from tradingflow.views import ArrayView
from ...operator import Operator, Notify
from ...types import Array, Handle, NodeKind


@dataclass(slots=True)
class MeanPredictorState[T]:
    # Configuration parameters.
    num_stocks: int
    num_features: int
    fit_fn: Callable[[np.ndarray, np.ndarray], T]
    predict_fn: Callable[["MeanPredictorState[T]", np.ndarray, T], np.ndarray]
    rebalance_period: int
    max_samples: int

    # Ticks elapsed from the last rebalance.
    tick_count: int = 0

    # Cross-sectional snapshots stored as Python lists of numpy arrays
    # (no copy on append).
    # features_list[t]: (num_stocks, num_features) array
    # prices_list[t]:   (num_stocks,) array
    features_list: list[np.ndarray] = field(default_factory=list)
    prices_list: list[np.ndarray] = field(default_factory=list)

    # Flat list of valid (t, s) sample indices, built incrementally.
    # Each entry is (time_index, stock_index) where features at time_index
    # are finite and both prices at time_index and time_index + k are finite.
    valid_indices: list[tuple[int, int]] = field(default_factory=list)

    # Random generator for uniform sampling of regression samples.
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


class MeanPredictor[T](
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Array[np.float64]]],
        Handle[Array[np.float64]],
        MeanPredictorState[T],
    ]
):
    """Abstract mean-return predictor.

    Runs on every tick.  Accumulates per-stock feature and price history
    in a ring buffer, and every ``rebalance_period`` ticks collects
    regression samples, calls the model's ``fit`` and ``predict``
    functions, and outputs predicted returns.  On non-rebalance ticks it
    returns ``False`` to halt downstream propagation.

    Subclasses must set ``_fit_fn`` and ``_predict_fn`` on the
    [`MeanPredictorState`] in their ``init`` override.

    Parameters
    ----------
    features
        Stacked features, shape ``(num_stocks, num_features)``.

    adjusted_prices
        Stacked forward-adjusted close prices, shape ``(num_stocks,)``.

    fit_fn
        ``(x, y) -> coefficients``.  Called with the design matrix
        ``x`` of shape ``(m, num_features)`` and target vector ``y`` of
        shape ``(m,)``.  Returns model parameters.

    predict_fn
        ``(state, features, parameters) -> predicted``.  Called with the
        current state, the latest feature matrix of shape
        ``(num_stocks, num_features)``, and the fitted parameters.
        Returns predicted returns of shape ``(num_stocks,)``.

    rebalance_period
        Produce output every N ticks.

    max_samples
        Maximum regression samples.
    """

    def __init__(
        self,
        features: Handle,
        adjusted_prices: Handle,
        *,
        fit_fn: Callable[[np.ndarray, np.ndarray], T],
        predict_fn: Callable[[MeanPredictorState[T], np.ndarray, T], np.ndarray],
        rebalance_period: int,
        max_samples: int,
    ) -> None:
        assert len(features.shape) == 2, "features must be a 2D array of shape (num_stocks, num_features)"
        assert len(adjusted_prices.shape) == 1, "prices must be a 1D array of shape (num_stocks,)"
        assert features.shape[0] == adjusted_prices.shape[0], "mismatched num_stocks between features and prices"

        self._num_stocks = features.shape[0]
        self._num_features = features.shape[-1]
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._rebalance_period = rebalance_period
        self._max_samples = max_samples

        super().__init__(
            inputs=(features, adjusted_prices),
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
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        notify: Notify,
    ) -> bool:
        N = state.num_stocks
        F = state.num_features
        K = state.rebalance_period
        T = len(state.features_list)

        # Append snapshot (store array reference, no copy).
        features = inputs[0].value()
        prices = inputs[1].value()
        state.features_list.append(features)
        state.prices_list.append(prices)

        # Incrementally update valid-index list.
        # The new tick at index T enables samples at time (T - K) whose
        # forward return endpoint is now known.
        if T >= K:
            t = T - K
            prev_features = state.features_list[t]  # (N, F)
            prev_prices = state.prices_list[t]  # (N,)
            feat_ok = np.isfinite(prev_features).all(axis=1)
            price_ok = np.isfinite(prev_prices) & (prev_prices > 0) & np.isfinite(prices) & (prices > 0)
            valid_stocks = np.where(feat_ok & price_ok)[0]
            for s in valid_stocks:
                state.valid_indices.append((t, s))

        # Only produce output every rebalance_period ticks.
        state.tick_count += 1
        if state.tick_count < state.rebalance_period:
            return False
        state.tick_count = 0

        # Need at least F + 2 samples to fit an intercept + F coefficients and have >1 degree of freedom.
        M = len(state.valid_indices)
        if M < F + 2:
            return False

        # Uniformly subsample up to `max_samples` from the valid indices.
        m = min(M, state.max_samples)
        if M > state.max_samples:
            chosen_indices = state.rng.choice(M, m, replace=False)
        else:
            chosen_indices = np.arange(M)

        # Build feature matrix and target vector for the chosen samples.
        x = np.empty((m, F), dtype=np.float64)
        y = np.empty(m, dtype=np.float64)
        for i, chosen_index in enumerate(chosen_indices):
            t, s = state.valid_indices[chosen_index]
            x[i] = state.features_list[t][s]
            y[i] = np.log(state.prices_list[t + K][s]) - np.log(state.prices_list[t][s])

        # Fit model and predict only for stocks with valid current features.
        params = MeanPredictor._fit(state, x, y)
        feat_ok = np.isfinite(features).all(axis=1)
        predicts = np.zeros(N, dtype=np.float64)
        predicts[feat_ok] = MeanPredictor._predict(state, features[feat_ok], params)
        output.write(predicts)
        return True

    @staticmethod
    def _fit(state: MeanPredictorState[T], x: np.ndarray, y: np.ndarray) -> T:
        return state.fit_fn(x, y)

    @staticmethod
    def _predict(state: MeanPredictorState[T], features: np.ndarray, params: T) -> np.ndarray:
        return state.predict_fn(state, features, params)
