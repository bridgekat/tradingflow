"""Abstract variance (covariance matrix) predictor operator."""

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator, SeriesView


@dataclass(slots=True)
class VariancePredictorState[T]:
    num_stocks: int
    num_features: int
    universe_size: int
    max_periods: int | None
    min_periods: int | None
    fit_fn: Callable[[np.ndarray, np.ndarray], T]
    predict_fn: Callable[["VariancePredictorState[T]", np.ndarray, T], np.ndarray]


class VariancePredictor[T](
    Operator[
        ArrayView[np.float64],
        SeriesView[np.float64],
        SeriesView[np.float64],
        ArrayView[np.float64],
        VariancePredictorState[T],
    ]
):
    """Abstract covariance matrix predictor.

    On every upstream tick, the predictor is invoked so subclasses can
    observe each new sample (a future incremental-fit hook can accumulate
    running statistics here — the current base class refits from scratch
    on rebalance, so non-rebalance ticks simply return without work).
    On each **rebalance** tick (signalled by the `universe` input
    producing new weights), reads the last `max_periods` feature and
    price entries from the upstream `Series` inputs, builds a 1-period
    return matrix, calls `fit_fn` and `predict_fn`, and emits the
    predicted covariance matrix.

    The rebalance cadence is controlled by the caller: typically
    `universe` is clocked by a rebalance clock (e.g. via
    [`Clocked`][tradingflow.operators.Clocked]), so universe updates
    coincide with rebalance dates.

    ## NaN behavior

    The emitted `(num_stocks, num_stocks)` covariance matrix may contain
    `NaN` rows and columns for stocks that are out of the universe, have
    non-finite features at the rebalance timestamp, or have fewer than
    `min_periods` valid historical observations.  The finite submatrix
    (indexed by the remaining stocks) is the output of `predict_fn` on a
    fully-masked feature subset — so `predict_fn` itself never needs to
    handle `NaN`.  Downstream portfolio constructors must accept `NaN`
    rows/columns and subset to the finite ones (see
    [`VariancePortfolio`][tradingflow.operators.portfolios.VariancePortfolio]
    and
    [`MeanVariancePortfolio`][tradingflow.operators.portfolios.MeanVariancePortfolio]).

    Parameters
    ----------
    universe
        Universe weights, shape `(num_stocks,)`.  Updates on this input
        trigger a rebalance.
    features_series
        Recorded features series, element shape
        `(num_stocks, num_features)`.
    adjusted_prices_series
        Recorded forward-adjusted close prices series, element shape
        `(num_stocks,)`.
    fit_fn
        `(x, y) -> params`.  Feature array `x` of shape
        `(T, N, F)` and 1-period return matrix `y` of shape
        `(T, N)`.
    predict_fn
        `(state, features, params) -> covariances`.  Current
        features of shape `(N, F)` and fitted params.
        `state.universe_size` gives the maximum number of stocks
        in the universe.
    universe_size
        Upper bound on the number of nonzero entries in the universe
        array.  Passed through to `predict_fn` via state for
        pre-allocation.
    max_periods
        Maximum number of most-recent time rows to feed to
        `fit_fn`.  `None` uses all available history.
    min_periods
        Minimum number of valid observations per stock.  Stocks with
        fewer valid (all-finite features and finite return) observations
        across the time rows receive `NaN` for their variance
        and all covariances with other stocks.  `None` disables
        per-stock filtering.
    """

    def __init__(
        self,
        universe: Handle,
        features_series: Handle,
        adjusted_prices_series: Handle,
        *,
        fit_fn: Callable[[np.ndarray, np.ndarray], T],
        predict_fn: Callable[[VariancePredictorState[T], np.ndarray, T], np.ndarray],
        universe_size: int,
        max_periods: int | None = None,
        min_periods: int | None = None,
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
        self._max_periods = max_periods
        self._min_periods = min_periods

        super().__init__(
            inputs=(universe, features_series, adjusted_prices_series),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_stocks, self._num_stocks),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[
            ArrayView[np.float64],
            SeriesView[np.float64],
            SeriesView[np.float64],
        ],
        timestamp: int,
    ) -> VariancePredictorState[T]:
        return VariancePredictorState(
            num_stocks=self._num_stocks,
            num_features=self._num_features,
            universe_size=self._universe_size,
            fit_fn=self._fit_fn,
            predict_fn=self._predict_fn,
            max_periods=self._max_periods,
            min_periods=self._min_periods,
        )

    @staticmethod
    def compute(
        state: VariancePredictorState[T],
        inputs: tuple[
            ArrayView[np.float64],
            SeriesView[np.float64],
            SeriesView[np.float64],
        ],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Emit only on rebalance ticks (signalled by `universe`
        # producing new weights).  See MeanPredictor for rationale.
        if not produced[0]:
            return False

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
        sigma = np.full((state.num_stocks, state.num_stocks), np.nan, dtype=np.float64)
        if M > 0 and N > 0:
            sigma[np.ix_(mask, mask)] = state.predict_fn(state, features[mask], state.fit_fn(x, y))

        output.write(sigma)
        return True
