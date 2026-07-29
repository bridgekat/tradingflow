"""Pooled OLS coefficients of a target scalar series on a baseline vector series."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegressionCoefficientsState:
    num_features: int
    max_periods: int | None
    min_periods: int | None


class RegressionCoefficients:
    r"""Pooled OLS regression coefficients on a signal cadence.

    On every signal tick, fits the multivariate linear model
    \(y = X \beta + \alpha\) by ordinary least squares using all
    available aligned ``(target, baseline)`` pairs recorded so far, and
    emits the coefficient vector of shape `(F + 1,)` with the intercept
    \(\alpha\) in the **last** position - matching the convention of
    the `LinearRegression` mean predictor.

    Solved with `np.linalg.lstsq` (SVD-based) so the operator is robust
    to rank deficiency: when the effective rank of the augmented design
    matrix is less than `F + 1` (under-sampled or collinear columns),
    the operator emits `NaN` rather than a minimum-norm solution.

    The intercept column is added internally, so the upstream baseline
    series should *not* include an all-ones column.

    Notes
    -----
    The two recorded series must be in lock-step at every signal tick:
    the operator asserts ``len(target) == len(baseline)`` on each fire,
    matching the alignment contract of the `MeanPredictor`.
    Use `ResampleSignalled` upstream to fold heterogeneous cadences down onto a
    single recording pulse.

    Non-finite samples (any NaN in the target or in any baseline column)
    are dropped before fitting.

    Inputs
    ------
    signal : None
        Unit/signal source.  The operator refits and emits only on
        signal ticks; the recorded series can update at any cadence in
        between.
    target : Series, element shape ``()``
        Recorded target series (scalar per tick).
    baseline : Series, element shape ``(F,)``
        Recorded baseline series.

    Parameters
    ----------
    num_features
        Number of baseline columns `F`.  Output has shape `(F + 1,)`.
    max_periods
        Maximum number of most-recent ``(target, baseline)`` pairs to
        feed into the fit.  `None` (default) uses every pair recorded so
        far.
    min_periods
        Minimum number of valid observations (after dropping rows where
        the target or any baseline column is non-finite) required for
        the operator to emit.  When fewer than `min_periods` valid
        observations are available at a signal tick, the operator
        returns without emitting.  `None` (default) disables the gate.
    """

    def __init__(
        self,
        num_features: int,
        *,
        max_periods: int | None = None,
        min_periods: int | None = None,
    ) -> None:
        assert max_periods is None or max_periods >= 1, "max_periods must be >= 1"
        assert min_periods is None or min_periods >= 1, "min_periods must be >= 1"

        self._num_features = num_features
        self._max_periods = max_periods
        self._min_periods = min_periods

    def init(self, inputs) -> RegressionCoefficientsState:
        return RegressionCoefficientsState(
            num_features=self._num_features,
            max_periods=self._max_periods,
            min_periods=self._min_periods,
        )

    @staticmethod
    def compute(
        inputs,
        state: RegressionCoefficientsState,
        timestamp: int,
    ) -> np.ndarray | None:
        # Refit only when the leading signal ticks.
        signal, target_view, baseline_view = inputs
        if not signal:
            return None

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
            return None

        if valid_count < n_params:
            return nan_out

        # Append intercept column at the right edge - intercept lands in
        # the last position of the coefficient vector.
        x_panel = np.column_stack([x_panel, np.ones(valid_count)])

        coef, _, rank, _ = np.linalg.lstsq(x_panel, y, rcond=None)

        if rank < n_params or not np.all(np.isfinite(coef)):
            return nan_out

        return coef


def build(**kwargs) -> RegressionCoefficients:
    """Construct a :class:`RegressionCoefficients` operator.

    Build kwargs
    ------------
    num_features : int
        Number of baseline columns `F`.  Output has shape `(F + 1,)`.
    max_periods : int | None, optional
        Trailing-window cap on recorded pairs.  Default `None`.
    min_periods : int | None, optional
        Firing gate on valid observations.  Default `None`.
    """
    return RegressionCoefficients(
        num_features=int(kwargs["num_features"]),
        max_periods=kwargs.get("max_periods"),
        min_periods=kwargs.get("min_periods"),
    )
