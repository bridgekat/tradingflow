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
    target_delay: int
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
    r"""Abstract cross-sectional covariance predictor.

    A `VariancePredictor` is a panel covariance estimator: on each
    **rebalance** tick (signalled by the `universe` input producing new
    weights), it reads the last `max_periods` aligned feature/target
    rows from the upstream `Series` inputs, calls `fit_fn` and
    `predict_fn`, and emits the predicted covariance matrix.  Non-
    rebalance ticks are ignored.

    The estimator itself is agnostic to what the target represents — linear
    returns, log returns, a rank-transformed return, a custom signal —
    so the **meaning of the prediction is defined by how the target
    series is constructed upstream**.  Typical targets are built from
    [`PctChange`][tradingflow.operators.num.pct_change.PctChange] (linear
    returns) or [`Log`][tradingflow.operators.num.arithmetic.Log] followed
    by [`Diff`][tradingflow.operators.num.diff.Diff] (log returns).

    ## Input data alignment

    At every rebalance, the invariant

        len(features_series) == len(target_series) + target_delay

    is asserted.  `features_series[i]` is then paired with
    `target_series[i]`, and the most recent `target_delay` rows of the
    features series are ignored in fitting.  This is useful when the
    target is a forward-looking returns series, which only becomes
    available *after* the features are observed.

    ## NaN behavior

    The emitted `(num_stocks, num_stocks)` covariance matrix may contain
    `NaN` rows and columns for stocks that are out of the universe, have
    non-finite features at the rebalance timestamp, or have fewer than
    `min_periods` valid historical observations.  The finite submatrix
    (indexed by the remaining stocks) is the output of `predict_fn` on a
    fully-masked feature subset — so `predict_fn` itself never needs to
    handle `NaN`.  Downstream portfolio constructors must accept `NaN`
    rows/columns and subset to the finite ones (see
    [`VariancePortfolio`][tradingflow.operators.portfolios.variance_portfolio.VariancePortfolio]
    and
    [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio]).

    Parameters
    ----------
    universe
        Universe weights, shape `(num_stocks,)`.  Updates on this input
        trigger a rebalance.
    features_series
        Recorded features series, element shape
        `(num_stocks, num_features)`.
    target_series
        Recorded target series, element shape `(num_stocks,)`.  The
        covariance matrix is estimated for cross-sectional samples of
        this series.
    fit_fn
        `(x, y) -> params`.  Feature array `x` of shape `(T, N, F)` and
        target matrix `y` of shape `(T, N)`.
    predict_fn
        `(state, features, params) -> covariances`.  Current features
        of shape `(N, F)` and fitted params.  `state.universe_size`
        gives the maximum number of stocks in the universe.
    universe_size
        Upper bound on the number of nonzero entries in the universe
        array.  Passed through to `predict_fn` via state for
        pre-allocation.
    target_delay
        Number of periods the target series lags the features series.
    max_periods
        Maximum number of most-recent time rows to feed to `fit_fn`.
        `None` uses all available aligned pairs.
    min_periods
        Minimum number of valid observations per stock.  Stocks with
        fewer valid (all-finite features and finite target)
        observations across the aligned time rows receive `NaN` for
        their variance and all covariances with other stocks.  `None`
        disables per-stock filtering.
    """

    def __init__(
        self,
        universe: Handle,
        features_series: Handle,
        target_series: Handle,
        *,
        fit_fn: Callable[[np.ndarray, np.ndarray], T],
        predict_fn: Callable[[VariancePredictorState[T], np.ndarray, T], np.ndarray],
        universe_size: int,
        target_delay: int,
        max_periods: int | None = None,
        min_periods: int | None = None,
    ) -> None:
        num_stocks, num_features = features_series.shape

        assert universe.shape == (num_stocks,)
        assert features_series.shape == (num_stocks, num_features)
        assert target_series.shape == (num_stocks,)
        assert target_delay >= 0

        self._num_stocks = num_stocks
        self._num_features = num_features
        self._universe_size = universe_size
        self._target_delay = target_delay
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._max_periods = max_periods
        self._min_periods = min_periods

        super().__init__(
            inputs=(universe, features_series, target_series),
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
            target_delay=self._target_delay,
            max_periods=self._max_periods,
            min_periods=self._min_periods,
            fit_fn=self._fit_fn,
            predict_fn=self._predict_fn,
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
        universe_view, features_series_view, target_series_view = inputs
        universe_produced, _, _ = produced

        # Emit only on rebalance ticks (signalled by the `universe`
        # input producing new weights).
        if not universe_produced:
            return False

        # Check data alignment.
        n_features_total = len(features_series_view)
        n_target = len(target_series_view)

        assert n_features_total == n_target + state.target_delay, (
            f"variance_predictor: input data does not align, "
            f"len(features_series)={n_features_total}, "
            f"len(target_series)={n_target}, "
            f"target_delay={state.target_delay}. "
            f"Expected len(features) == len(target) + target_delay."
        )

        # Training window.
        n_use = min(n_target, state.max_periods) if state.max_periods is not None else n_target
        start = n_target - n_use

        all_features = features_series_view.values(start, start + n_use)  # (T, N, F)
        all_target = target_series_view.values(start, start + n_use)  # (T, N)

        # Per-stock valid observation counts.
        valid = np.isfinite(all_features).all(axis=2) & np.isfinite(all_target)
        counts = valid.sum(axis=0)

        # Current features for prediction.
        features = features_series_view[-1]

        # Universe filter.
        mask = universe_view.to_numpy() > 0
        assert (
            int(mask.sum()) <= state.universe_size
        ), f"variance_predictor: universe has {int(mask.sum())} nonzero entries, exceeds universe_size={state.universe_size}"

        if state.min_periods is not None:
            mask &= counts >= state.min_periods

        # Filter to stocks with valid features for prediction.
        mask &= np.isfinite(features).all(axis=1)

        x = all_features[:, mask, :]  # (T, N, F)
        y = all_target[:, mask]  # (T, N)

        sigma = np.full((state.num_stocks, state.num_stocks), np.nan, dtype=np.float64)
        if n_use > 0 and mask.any():
            sigma[np.ix_(mask, mask)] = state.predict_fn(state, features[mask], state.fit_fn(x, y))

        output.write(sigma)
        return True
