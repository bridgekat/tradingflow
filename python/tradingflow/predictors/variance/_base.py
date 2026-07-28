"""Abstract variance (covariance matrix) predictor operator (pyhost)."""

from typing import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class VariancePredictorState[T]:
    num_stocks: int
    num_features: int
    universe_size: int
    target_offset: int
    refit_every: int
    max_periods: int | None
    min_periods: int | None
    fit_fn: Callable[[np.ndarray, np.ndarray], T]
    predict_fn: Callable[["VariancePredictorState[T]", np.ndarray, T], np.ndarray]
    # Updated across compute() calls - the fitted params from the last
    # refit tick are reused until the next cadence-driven refit.
    # `fitted` gates prediction: it flips to True after the first
    # successful fit.  `cached_params` may legitimately be `None` after
    # a fit (e.g., pass-through predictors that don't fit any params).
    cached_params: T | None = None
    fitted: bool = False
    rebalance_count: int = 0


class VariancePredictor[T]:
    r"""Abstract cross-sectional covariance predictor.

    A `VariancePredictor` is a panel covariance estimator: on each
    **rebalance** tick (signalled by the `universe` input producing new
    weights), it reads the last `max_periods` aligned feature/target
    pairs from the upstream `Series` inputs, calls `fit_fn` and
    `predict_fn`, and emits the predicted covariance matrix.  Non-
    rebalance ticks are ignored.

    Input ports (in order): ``(universe, features_series, target_series)``

    - universe: ArrayView, shape ``(num_stocks,)``.  Updates trigger a
      rebalance.
    - features_series: SeriesView, element shape ``(num_stocks, num_features)``.
    - target_series: SeriesView, element shape ``(num_stocks,)``.  Must be
      the same length as ``features_series``.

    Output: ArrayView, shape ``(num_stocks, num_stocks)`` (covariance).

    Config (constructor / build kwargs)
    -----------------------------------
    num_stocks, num_features : int
        Panel dimensions.  Derived from input handles in the legacy
        operator; passed as build kwargs here.
    fit_fn : (x, y) -> params
        ``x`` of shape ``(T, N, F)`` and ``y`` of shape ``(T, N)``.
    predict_fn : (state, features, params) -> covariances
        Current features ``(N, F)`` and fitted params.
    universe_size : int
        Upper bound on the number of nonzero entries in ``universe``.
    target_offset : int
        Non-negative forward offset pairing ``features[i]`` with
        ``target[i + target_offset]``.
    refit_every : int
        Refit cadence in rebalance ticks (default ``1``).
    max_periods : int | None
        Maximum number of most-recent ``(feature, target)`` pairs to feed
        to ``fit_fn``.  ``None`` uses all available pairs.
    min_periods : int | None
        Minimum number of valid observations per stock.  ``None`` disables
        per-stock filtering.
    """

    def __init__(
        self,
        *,
        num_stocks: int,
        num_features: int,
        fit_fn: Callable[[np.ndarray, np.ndarray], T],
        predict_fn: Callable[[VariancePredictorState[T], np.ndarray, T], np.ndarray],
        universe_size: int,
        target_offset: int,
        refit_every: int = 1,
        max_periods: int | None = None,
        min_periods: int | None = None,
    ) -> None:
        assert target_offset >= 0
        assert refit_every >= 1

        self._num_stocks = int(num_stocks)
        self._num_features = int(num_features)
        self._universe_size = int(universe_size)
        self._target_offset = int(target_offset)
        self._refit_every = int(refit_every)
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._max_periods = max_periods
        self._min_periods = min_periods

    def init(self, inputs) -> VariancePredictorState[T]:
        return VariancePredictorState(
            num_stocks=self._num_stocks,
            num_features=self._num_features,
            universe_size=self._universe_size,
            target_offset=self._target_offset,
            refit_every=self._refit_every,
            max_periods=self._max_periods,
            min_periods=self._min_periods,
            fit_fn=self._fit_fn,
            predict_fn=self._predict_fn,
        )

    @staticmethod
    def compute(
        inputs,
        state: VariancePredictorState[T],
        timestamp: int,
    ) -> np.ndarray | None:
        rebalance, universe_view, features_series_view, target_series_view = inputs

        # Emit only on rebalance ticks (the leading clock pulses).
        if not rebalance:
            return None

        # Check data alignment - features and target must tick in lock-step.
        n_features = len(features_series_view)
        n_target = len(target_series_view)

        assert n_features == n_target, (
            f"variance_predictor: features and target lengths differ, "
            f"len(features_series)={n_features}, "
            f"len(target_series)={n_target}. "
            f"Expected equal lengths."
        )

        # Refit if we've never fit, or if this rebalance hits the cadence.
        # Otherwise reuse the cached parameters from the last refit.
        should_refit = (not state.fitted) or (state.rebalance_count % state.refit_every == 0)
        state.rebalance_count += 1

        # Available (feature, target) training pairs: features[i] paired
        # with target[i + target_offset] for i in 0..n_pair.
        n_pair = max(0, n_target - state.target_offset)

        # Training window: last n_use pairs.
        n_use = min(n_pair, state.max_periods) if state.max_periods is not None else n_pair
        start = n_pair - n_use

        all_features = features_series_view.values(start, start + n_use)  # (T, N, F)
        all_target = target_series_view.values(
            start + state.target_offset, start + state.target_offset + n_use
        )  # (T, N)

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

        # Refit the model when the cadence fires; otherwise keep cached params.
        if should_refit and n_use > 0 and mask.any():
            x = all_features[:, mask, :]  # (T, N, F)
            y = all_target[:, mask]  # (T, N)
            state.cached_params = state.fit_fn(x, y)
            state.fitted = True

        sigma = np.full((state.num_stocks, state.num_stocks), np.nan, dtype=np.float64)
        if state.fitted and mask.any():
            sigma[np.ix_(mask, mask)] = state.predict_fn(state, features[mask], state.cached_params)

        return sigma
