"""Abstract mean-return predictor operator."""

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ...views import ArrayView, SeriesView
from ...operator import Operator, Notify
from ...types import Array, Series, Handle, NodeKind
from ...utils import coerce_timestamp


@dataclass(slots=True)
class MeanPredictorState[T]:
    num_stocks: int
    num_features: int
    universe_size: int
    rebalance_periods: int
    max_periods: int | None
    min_periods: int | None
    trading_start: int | None
    fit_fn: Callable[[np.ndarray, np.ndarray], T]
    predict_fn: Callable[["MeanPredictorState[T]", np.ndarray, T], np.ndarray]

    tick_count: int = 0


class MeanPredictor[T](
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Series[np.float64]], Handle[Series[np.float64]]],
        Handle[Array[np.float64]],
        MeanPredictorState[T],
    ]
):
    """Abstract mean-return predictor.

    Runs on every tick.  Every ``rebalance_periods`` ticks after
    ``trading_start``, reads the last ``max_periods`` feature and
    price entries from upstream ``Series`` inputs, builds a
    **1-period** return matrix (at sampling frequency), calls
    ``fit_fn`` and ``predict_fn``, and outputs predicted returns.
    The first tick after ``trading_start`` always triggers a rebalance.

    Parameters
    ----------
    universe
        Universe weights, shape ``(num_stocks,)``.
    features_series
        Recorded features series, element shape
        ``(num_stocks, num_features)``.
    adjusted_prices_series
        Recorded forward-adjusted close prices series, element shape
        ``(num_stocks,)``.
    fit_fn
        ``(x, y) -> params``.  Feature array ``x`` of shape
        ``(T, N, F)`` and 1-period return matrix ``y`` of shape
        ``(T, N)``.
    predict_fn
        ``(state, features, params) -> returns``.  Current features
        of shape ``(N, F)`` and fitted params.  ``state.universe_size``
        gives the maximum number of stocks in the universe.
    universe_size
        Upper bound on the number of nonzero entries in the universe
        array.  Passed through to ``predict_fn`` via state for
        pre-allocation.
    rebalance_periods
        Produce output every N ticks (controls refit cadence, not
        the prediction target horizon).
    max_periods
        Maximum number of most-recent time rows to feed to
        ``fit_fn``.  ``None`` uses all available history.
    min_periods
        Minimum number of valid observations per stock.  Stocks with
        fewer valid (all-finite features and finite return) observations
        across the time rows receive ``NaN`` in the output.  ``None``
        disables per-stock filtering.
    trading_start
        If set, suppress output before this timestamp.  The first
        tick at or after this timestamp always triggers a rebalance.
    """

    def __init__(
        self,
        universe: Handle,
        features_series: Handle,
        adjusted_prices_series: Handle,
        *,
        fit_fn: Callable[[np.ndarray, np.ndarray], T],
        predict_fn: Callable[[MeanPredictorState[T], np.ndarray, T], np.ndarray],
        universe_size: int,
        rebalance_periods: int,
        max_periods: int | None = None,
        min_periods: int | None = None,
        trading_start: np.datetime64 | None = None,
    ) -> None:
        assert len(universe.shape) == 1
        assert len(features_series.shape) == 2
        assert len(adjusted_prices_series.shape) == 1
        assert universe.shape[0] == features_series.shape[0] == adjusted_prices_series.shape[0]

        self._num_stocks = features_series.shape[0]
        self._num_features = features_series.shape[-1]
        self._universe_size = universe_size
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._rebalance_periods = rebalance_periods
        self._max_periods = max_periods
        self._min_periods = min_periods
        self._trading_start = int(coerce_timestamp(trading_start)) if trading_start is not None else None

        super().__init__(
            inputs=(universe, features_series, adjusted_prices_series),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_stocks,),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> MeanPredictorState[T]:
        return MeanPredictorState(
            num_stocks=self._num_stocks,
            num_features=self._num_features,
            universe_size=self._universe_size,
            fit_fn=self._fit_fn,
            predict_fn=self._predict_fn,
            rebalance_periods=self._rebalance_periods,
            max_periods=self._max_periods,
            min_periods=self._min_periods,
            trading_start=self._trading_start,
        )

    @staticmethod
    def compute(
        state: MeanPredictorState[T],
        inputs: tuple[ArrayView[np.float64], SeriesView[np.float64], SeriesView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        notify: Notify,
    ) -> bool:
        # Changes in universe only should not trigger recomputation.
        if not notify.input_produced()[1] and not notify.input_produced()[2]:
            return False

        # Suppress output before trading start.
        if state.trading_start is not None and timestamp < state.trading_start:
            return False

        # Rebalance every rebalance_periods ticks after trading_start.
        # First tick after trading_start always triggers (tick_count == 0).
        if state.tick_count > 0 and state.tick_count < state.rebalance_periods:
            state.tick_count += 1
            return False
        state.tick_count = 1

        universe, features_series, prices_series = inputs

        # Bulk-read the last max_periods entries from both series.
        n_available = max(0, len(prices_series) - 1)
        n_use = min(n_available, state.max_periods) if state.max_periods is not None else n_available
        start = n_available - n_use

        all_features = features_series.values(start, start + n_use)  # (M, N, F)
        all_prices = prices_series.values(start, start + n_use + 1)  # (M+1, N)

        # Vectorized 1-period returns: (M, N).
        prices_curr = np.where(all_prices[:-1] > 0, all_prices[:-1], np.nan)
        prices_next = np.where(all_prices[1:] > 0, all_prices[1:], np.nan)
        all_returns = prices_next / prices_curr - 1.0

        # Per-stock valid observation counts.
        valid = np.isfinite(all_features).all(axis=2) & np.isfinite(all_returns)
        counts = valid.sum(axis=0)  # (N,)

        # Current features for prediction.
        features = features_series[-1]

        # Filter to stocks currently in the universe.
        mask = universe.to_numpy() > 0
        assert int(mask.sum()) <= state.universe_size, (
            f"universe has {int(mask.sum())} nonzero entries, " f"exceeds universe_size={state.universe_size}"
        )

        # Filter to stocks with enough valid observations.
        if state.min_periods is not None:
            mask &= counts >= state.min_periods

        # Filter to stocks with valid features for prediction.
        mask &= np.isfinite(features).all(axis=1)

        M, N, F = n_use, int(mask.sum()), state.num_features
        x = all_features[:, mask, :]  # (M, N, F)
        y = all_returns[:, mask]  # (M, N)

        # Fit and predict.
        mu = np.full((state.num_stocks,), np.nan, dtype=np.float64)
        if M > 0 and N > 0:
            mu[mask] = state.predict_fn(state, features[mask], state.fit_fn(x, y))

        output.write(mu)
        return True
