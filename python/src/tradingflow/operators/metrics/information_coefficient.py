"""Cross-sectional Information Coefficient (IC / RankIC) evaluator."""

from dataclasses import dataclass

import numpy as np

from ...views import ArrayView, SeriesView
from ...operator import Operator, Notify
from ...types import Array, Series, Handle, NodeKind
from ...utils import coerce_timestamp


@dataclass(slots=True)
class InformationCoefficientState:
    ranking: bool
    num_stocks: int
    trading_start: int | None
    initialized: bool = False
    predictions: np.ndarray | None = None
    sum_ic: float = 0.0
    count: int = 0


class InformationCoefficient(
    Operator[
        tuple[Handle[Series[np.float64]], Handle[Series[np.float64]]],
        Handle[Array[np.float64]],
        InformationCoefficientState,
    ]
):
    """Cross-sectional IC / RankIC evaluator.

    Evaluates mean-return prediction quality.  On each prediction
    emission, stores the predicted scores and begins accumulating
    daily cross-sectional correlations between the scores and
    realized 1-period returns.  When the next prediction arrives,
    emits the mean daily IC over the evaluation period and updates
    the stored scores.

    Output is a scalar (the mean daily cross-sectional IC over one
    evaluation period).  ``Record(output)`` produces a directly
    plottable time series.

    Alignment guarantee
    -------------------
    After the initial warmup (first prediction stores scores without
    emitting), the operator emits exactly once per
    ``predictions_series`` entry (NaN when data is unavailable).
    The prediction timestamp for ``record[i]`` is
    ``predictions_ts[-(n + 1):-1][i]`` where ``n`` is the number
    of recorded outputs.

    Parameters
    ----------
    predictions_series
        Recorded predicted scores series, element shape ``(N,)``.
        Typically ``Record(predicted_returns)`` or ``Record(factor)``.
    adjusted_prices_series
        Recorded forward-adjusted close prices series, element shape
        ``(N,)``.
    ranking
        If ``False`` (default), compute Pearson IC.  If ``True``,
        rank-transform both inputs first to compute Spearman RankIC.
    trading_start
        If set, suppress output before this timestamp.
    min_valid
        Minimum number of non-NaN cross-sectional pairs required to
        compute a valid daily IC.  Days below threshold are skipped.
    min_periods
        Minimum number of accumulated daily IC values required to
        emit a valid mean.  Emits NaN if below threshold.
    """

    def __init__(
        self,
        predictions_series: Handle,
        adjusted_prices_series: Handle,
        *,
        ranking: bool = False,
        trading_start: np.datetime64 | None = None,
        min_valid: int = 10,
        min_periods: int = 1,
    ) -> None:
        assert len(predictions_series.shape) == 1
        assert len(adjusted_prices_series.shape) == 1
        assert predictions_series.shape[0] == adjusted_prices_series.shape[0]

        self._ranking = ranking
        self._num_stocks = predictions_series.shape[0]
        self._trading_start = int(coerce_timestamp(trading_start)) if trading_start is not None else None

        super().__init__(
            inputs=(predictions_series, adjusted_prices_series),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> InformationCoefficientState:
        return InformationCoefficientState(
            ranking=self._ranking,
            num_stocks=self._num_stocks,
            trading_start=self._trading_start,
        )

    @staticmethod
    def compute(
        state: InformationCoefficientState,
        inputs: tuple[SeriesView[np.float64], SeriesView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        notify: Notify,
    ) -> bool:
        predictions_series, prices_series = inputs
        predictions_produced, prices_produced = notify.input_produced()

        # Accumulate one-period IC on price ticks.
        if prices_produced and state.predictions is not None and len(prices_series) >= 2:
            prices_old = np.where(prices_series[-2] > 0, prices_series[-2], np.nan)
            prices_new = np.where(prices_series[-1] > 0, prices_series[-1], np.nan)
            r = prices_new / prices_old - 1.0
            s = state.predictions
            valid = np.isfinite(s) & np.isfinite(r)
            s, r = s[valid], r[valid]
            if state.ranking:
                s, r = _rank(s), _rank(r)
            if len(s) >= 2:
                ic = float(np.corrcoef(s, r)[0, 1])
                state.sum_ic += ic
            state.count += 1

        # Gate: new prediction?
        if not predictions_produced:
            return False

        # Suppress output before trading start.
        if state.trading_start is not None and timestamp < state.trading_start:
            return False

        # First prediction stores scores without emitting.
        if not state.initialized:
            state.initialized = True
            state.predictions = predictions_series[-1]
            return False

        # Emit mean daily IC over the evaluation period.
        mean_ic = state.sum_ic / max(state.count, 1)
        output.write(np.array(mean_ic, dtype=np.float64))

        # Update stored predictions and reset accumulators.
        state.predictions = predictions_series[-1]
        state.sum_ic = 0.0
        state.count = 0

        return True


def _rank(x: np.ndarray) -> np.ndarray:
    """Rank 1-D array with average tie-breaking (no scipy dependency).

    Assumes all values are finite.
    """
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Average ranks for tied values.
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        if j > i + 1:
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j
    return ranks
