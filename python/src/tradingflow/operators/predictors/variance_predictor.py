"""Abstract variance (covariance matrix) predictor operator."""

from typing import Callable
from dataclasses import dataclass, field

import numpy as np

from ...views import ArrayView, SeriesView
from ...operator import Operator, Notify
from ...types import Array, Series, Handle, NodeKind


@dataclass(slots=True)
class VariancePredictorState[T]:
    num_stocks: int
    num_features: int
    rebalance_period: int
    max_samples: int
    min_samples: int
    fit_fn: Callable[[np.ndarray, np.ndarray], T]
    predict_fn: Callable[["VariancePredictorState[T]", np.ndarray, T], np.ndarray]

    tick_count: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


class VariancePredictor[T](
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Series[np.float64]], Handle[Series[np.float64]]],
        Handle[Array[np.float64]],
        VariancePredictorState[T],
    ]
):
    """Abstract covariance matrix predictor.

    Runs on every tick.  Every ``rebalance_period`` ticks, reads the
    accumulated feature and price history from upstream ``Series``
    inputs, builds a return matrix, calls ``fit_fn`` and ``predict_fn``,
    and outputs the predicted covariance matrix.

    Parameters
    ----------
    universe
        Universe weights, shape ``(num_stocks,)``.
    features_series
        Recorded features series, element shape ``(num_stocks, num_features)``.
    adjusted_prices_series
        Recorded forward-adjusted close prices series, element shape
        ``(num_stocks,)``.
    fit_fn
        ``(x, y) -> params``.  Feature array ``x`` of shape
        ``(T, N, F)`` and return matrix ``y`` of shape ``(T, N)``.
    predict_fn
        ``(state, features, params) -> covariances``.  Current features
        of shape ``(N, F)`` and fitted params.
    rebalance_period
        Produce output every N ticks.
    max_samples
        Maximum number of time rows to feed to ``fit_fn``.
    min_samples
        Minimum number of valid observations per stock.  Stocks with
        fewer valid (all-finite features and finite return) observations
        across the sampled time rows receive ``NaN`` for their variance
        and all covariances with other stocks.
    """

    def __init__(
        self,
        universe: Handle,
        features_series: Handle,
        adjusted_prices_series: Handle,
        *,
        fit_fn: Callable[[np.ndarray, np.ndarray], T],
        predict_fn: Callable[[VariancePredictorState[T], np.ndarray, T], np.ndarray],
        rebalance_period: int,
        max_samples: int = 1000,
        min_samples: int = 2,
    ) -> None:
        assert len(universe.shape) == 1
        assert len(features_series.shape) == 2
        assert len(adjusted_prices_series.shape) == 1
        assert universe.shape[0] == features_series.shape[0] == adjusted_prices_series.shape[0]

        self._num_stocks = features_series.shape[0]
        self._num_features = features_series.shape[1]
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._rebalance_period = rebalance_period
        self._max_samples = max_samples
        self._min_samples = min_samples

        super().__init__(
            inputs=(universe, features_series, adjusted_prices_series),
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
            min_samples=self._min_samples,
            fit_fn=self._fit_fn,
            predict_fn=self._predict_fn,
        )

    @staticmethod
    def compute(
        state: VariancePredictorState[T],
        inputs: tuple[ArrayView[np.float64], SeriesView[np.float64], SeriesView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        notify: Notify,
    ) -> bool:
        # Changes in universe only should not trigger recomputation.
        if not notify.input_produced()[1] and not notify.input_produced()[2]:
            return False

        # Only produce output every rebalance_period ticks.
        state.tick_count += 1
        if state.tick_count < state.rebalance_period:
            return False
        state.tick_count = 0

        universe, features_series, prices_series = inputs

        # Build cross-sectional return matrix and feature array for
        # universe stocks, subsampling time rows if needed.
        m_periods = max(0, len(prices_series) - state.rebalance_period)

        # Subsample time rows if too many.
        if m_periods > state.max_samples:
            ts = state.rng.choice(m_periods, state.max_samples, replace=False)
            ts.sort()
        else:
            ts = np.arange(m_periods)

        # Collect features and returns for the sampled time rows.
        all_features = []
        all_returns = []
        counts = np.zeros((state.num_stocks,), dtype=np.int64)
        for t in ts:
            t = int(t)
            features = features_series[t]
            prices_curr = prices_series[t]
            prices_curr = np.where(prices_curr > 0, prices_curr, np.nan)
            prices_next = prices_series[t + state.rebalance_period]
            prices_next = np.where(prices_next > 0, prices_next, np.nan)
            returns = prices_next / prices_curr - 1.0

            all_features.append(features)
            all_returns.append(returns)
            counts += np.isfinite(features).all(axis=1) & np.isfinite(returns)

        # Current features for prediction.
        features = features_series[-1]

        # Filter to stocks currently in the universe.
        mask = universe.to_numpy() > 0

        # Filter to stocks with enough valid observations.
        mask &= counts >= state.min_samples

        # Filter to stocks with valid features for prediction.
        mask &= np.isfinite(features).all(axis=1)

        # Combine sampled time rows into arrays of shape (M, N, F) and (M, N).
        M, N, F = len(ts), mask.sum(), state.num_features
        x = np.empty((M, N, F), dtype=np.float64)
        y = np.empty((M, N), dtype=np.float64)
        for i, t in enumerate(ts):
            x[i] = all_features[i][mask]
            y[i] = all_returns[i][mask]

        # Fit and predict only for stocks with sufficient data.
        sigma = np.full((state.num_stocks, state.num_stocks), np.nan, dtype=np.float64)
        if M > 0 and N > 0:
            sigma[np.ix_(mask, mask)] = state.predict_fn(state, features[mask], state.fit_fn(x, y))

        output.write(sigma)
        return True
