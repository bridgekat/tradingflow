"""Pooled OLS coefficients of a target scalar series on a baseline vector series."""

from dataclasses import dataclass

import numpy as np

from .... import ArrayView, Handle, NodeKind, Operator, SeriesView


@dataclass(slots=True)
class RegressionCoefficientsState:
    num_features: int
    max_periods: int | None
    min_periods: int | None


class RegressionCoefficients(
    Operator[
        None,
        SeriesView[np.float64],
        SeriesView[np.float64],
        ArrayView[np.float64],
        RegressionCoefficientsState,
    ]
):
    r"""Pooled OLS regression coefficients on a clock cadence.

    On every clock tick, fits the multivariate linear model
    \(y = X \beta + \alpha\) by ordinary least squares using all
    available aligned ``(target, baseline)`` pairs recorded so far, and
    emits the coefficient vector of shape `(F + 1,)` with the intercept
    \(\alpha\) in the **last** position - matching the convention of
    [`LinearRegression`][tradingflow.operators.predictors.mean.linear_regression.LinearRegression].

    Solved with `np.linalg.lstsq` (SVD-based) so the operator is robust
    to rank deficiency: when the effective rank of the augmented design
    matrix is less than `F + 1` (under-sampled or collinear columns),
    the operator emits `NaN` rather than a minimum-norm solution.

    The intercept column is added internally, so the upstream baseline
    series should *not* include an all-ones column.

    Notes
    -----
    The two recorded series must be in lock-step at every clock tick:
    the operator asserts ``len(target) == len(baseline)`` on each fire,
    matching the alignment contract of
    [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor].
    Use [`Resample`][tradingflow.operators.resample.Resample] upstream
    to fold heterogeneous cadences down onto a single recording pulse.

    Non-finite samples (any NaN in the target or in any baseline column)
    are dropped before fitting.

    Parameters
    ----------
    clock
        Clock source handle.  The operator refits and emits only on
        clock ticks; the recorded series can update at any cadence in
        between.
    target
        Recorded target series with element shape `()` (scalar per
        tick).
    baseline
        Recorded baseline series with element shape `(F,)`.
    max_periods
        Maximum number of most-recent ``(target, baseline)`` pairs to
        feed into the fit.  `None` (default) uses every pair recorded so
        far.  Same role as
        [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor]'s
        `max_periods`.
    min_periods
        Minimum number of valid observations (after dropping rows where
        the target or any baseline column is non-finite) required for
        the operator to emit.  When fewer than `min_periods` valid
        observations are available at a clock tick, the operator
        returns without emitting (downstream sees no update).  `None`
        (default) disables the gate, in which case the operator always
        emits on a clock tick - falling back to NaN coefficients when
        the design matrix is undersampled or rank-deficient.
    """

    def __init__(
        self,
        clock: Handle,
        target: Handle,
        baseline: Handle,
        *,
        max_periods: int | None = None,
        min_periods: int | None = None,
    ) -> None:
        assert clock.kind == NodeKind.UNIT, "clock must be a Unit-kind handle"
        assert target.kind == NodeKind.SERIES, "target must be a Series handle"
        assert baseline.kind == NodeKind.SERIES, "baseline must be a Series handle"
        assert target.shape == (), f"target series must be scalar, got shape {target.shape}"
        assert len(baseline.shape) == 1, f"baseline series must be 1D, got shape {baseline.shape}"
        assert max_periods is None or max_periods >= 1, "max_periods must be >= 1"
        assert min_periods is None or min_periods >= 1, "min_periods must be >= 1"

        self._num_features = baseline.shape[0]
        self._max_periods = max_periods
        self._min_periods = min_periods

        super().__init__(
            inputs=(clock, target, baseline),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_features + 1,),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[None, SeriesView[np.float64], SeriesView[np.float64]],
        timestamp: int,
    ) -> RegressionCoefficientsState:
        return RegressionCoefficientsState(
            num_features=self._num_features,
            max_periods=self._max_periods,
            min_periods=self._min_periods,
        )

    @staticmethod
    def compute(
        state: RegressionCoefficientsState,
        inputs: tuple[None, SeriesView[np.float64], SeriesView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Refit only when the clock ticks.
        if not produced[0]:
            return False

        _, target_view, baseline_view = inputs

        n_target = len(target_view)
        n_baseline = len(baseline_view)
        assert n_target == n_baseline, (
            f"regression_coefficients: target and baseline series lengths differ, "
            f"len(target)={n_target}, len(baseline)={n_baseline}. "
            f"Expected equal lengths."
        )

        n_params = state.num_features + 1
        nan_out = np.full((n_params,), np.nan, dtype=np.float64)

        # Trailing window: take only the last n_use pairs.
        n_use = min(n_target, state.max_periods) if state.max_periods is not None else n_target
        start = n_target - n_use

        y = target_view.values(start, start + n_use)
        x_panel = baseline_view.values(start, start + n_use)

        valid = np.isfinite(y) & np.isfinite(x_panel).all(axis=1)
        y, x_panel = y[valid], x_panel[valid]
        valid_count = x_panel.shape[0]

        # min_periods firing gate: when set, hold off entirely until
        # enough valid observations have accumulated.
        if state.min_periods is not None and valid_count < state.min_periods:
            return False

        if valid_count < n_params:
            output.write(nan_out)
            return True

        # Append intercept column at the right edge - intercept lands in
        # the last position of the coefficient vector.
        x_panel = np.column_stack([x_panel, np.ones(valid_count)])

        coef, _, rank, _ = np.linalg.lstsq(x_panel, y, rcond=None)

        if rank < n_params or not np.all(np.isfinite(coef)):
            output.write(nan_out)
            return True

        output.write(coef)
        return True
