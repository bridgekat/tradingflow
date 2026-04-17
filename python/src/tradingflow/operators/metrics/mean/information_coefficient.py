"""Cross-sectional Information Coefficient (IC / RankIC) evaluator."""

from dataclasses import dataclass

import numpy as np

from ....views import ArrayView
from ....operator import Operator
from ....types import Array, Handle, NodeKind


@dataclass(slots=True)
class InformationCoefficientState:
    ranking: bool
    num_stocks: int
    initialized: bool = False
    predictions: np.ndarray | None = None
    prev_prices: np.ndarray | None = None
    sum_ic: float = 0.0
    count: int = 0


class InformationCoefficient(
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Array[np.float64]]],
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
    emitting), the operator emits exactly once per prediction
    emission (NaN when data is unavailable).

    Memory
    ------
    Both ``predictions`` and ``prices`` are ``Array`` inputs — the
    operator only reads the latest cross-section of each, caching
    the previous price tick in state to compute one-period returns.
    No ``Record`` is required upstream.

    Parameters
    ----------
    predictions
        Live predicted scores, shape ``(N,)``.  Typically a
        mean-return predictor output or a factor.
    prices
        Live forward-adjusted close prices, shape ``(N,)``.
    ranking
        If ``False`` (default), compute Pearson IC.  If ``True``,
        rank-transform both inputs first to compute Spearman RankIC.
    min_valid
        Minimum number of non-NaN cross-sectional pairs required to
        compute a valid daily IC.  Days below threshold are skipped.
    min_periods
        Minimum number of accumulated daily IC values required to
        emit a valid mean.  Emits NaN if below threshold.
    """

    def __init__(
        self,
        predictions: Handle,
        prices: Handle,
        *,
        ranking: bool = False,
        min_valid: int = 10,
        min_periods: int = 1,
    ) -> None:
        assert len(predictions.shape) == 1
        assert len(prices.shape) == 1
        assert predictions.shape[0] == prices.shape[0]

        self._ranking = ranking
        self._num_stocks = predictions.shape[0]

        super().__init__(
            inputs=(predictions, prices),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> InformationCoefficientState:
        return InformationCoefficientState(
            ranking=self._ranking,
            num_stocks=self._num_stocks,
        )

    @staticmethod
    def compute(
        state: InformationCoefficientState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        predictions, prices = inputs
        predictions_produced, prices_produced = produced

        # Accumulate one-period IC on price ticks.
        if prices_produced:
            prices_new = np.where(prices.value() > 0, prices.value(), np.nan)
            if state.predictions is not None and state.prev_prices is not None:
                r = prices_new / state.prev_prices - 1.0
                s = state.predictions
                valid = np.isfinite(s) & np.isfinite(r)
                s, r = s[valid], r[valid]
                if state.ranking:
                    s, r = _rank(s), _rank(r)
                if len(s) >= 2:
                    ic = float(np.corrcoef(s, r)[0, 1])
                    state.sum_ic += ic
                state.count += 1
            state.prev_prices = prices_new

        # Gate: new prediction?
        if not predictions_produced:
            return False

        # First prediction stores scores without emitting.
        if not state.initialized:
            state.initialized = True
            state.predictions = predictions.value()
            return False

        # Emit mean daily IC over the evaluation period.
        mean_ic = state.sum_ic / max(state.count, 1)
        output.write(np.array(mean_ic, dtype=np.float64))

        # Update stored predictions and reset accumulators.
        state.predictions = predictions.value()
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
