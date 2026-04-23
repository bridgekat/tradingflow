"""Cross-sectional Information Coefficient evaluator."""

from dataclasses import dataclass

import numpy as np

from .... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class InformationCoefficientState:
    num_stocks: int
    initialized: bool = False
    predictions: np.ndarray | None = None
    sum_ic: float = 0.0
    count: int = 0


class InformationCoefficient(
    Operator[
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        InformationCoefficientState,
    ]
):
    """Cross-sectional IC evaluator.

    Evaluates prediction quality against a realized cross-sectional
    target.  On each prediction emission, stores the predicted scores
    and begins accumulating daily cross-sectional correlations between
    the scores and the realized target ticks that follow.  When the
    next prediction arrives, emits the mean daily IC over the
    evaluation period and updates the stored scores.

    Output is a scalar — the mean daily cross-sectional correlation
    over one evaluation period.  `Record(output)` produces a directly
    plottable time series.

    Notes
    -----
    **Alignment guarantee.** After the initial warmup (first prediction
    stores scores without emitting), the operator emits exactly once per
    prediction emission (NaN when data is unavailable).

    **Memory.** Both `predictions` and `target` are `Array` inputs —
    the operator only reads the latest cross-section of each.
    No `Record` is required upstream.

    Parameters
    ----------
    predictions
        Live predicted scores, shape `(N,)`.  Typically a mean-return
        predictor output or a factor.
    target
        Live cross-sectional target values, shape `(N,)`, produced at
        every tick (e.g. a `PctChange` node, optionally passed through
        `Gaussianize` / `Percentile` / etc.).
    """

    def __init__(self, predictions: Handle, target: Handle) -> None:
        num_stocks = predictions.shape[0]

        assert predictions.shape == (num_stocks,)
        assert target.shape == (num_stocks,)

        self._num_stocks = num_stocks

        super().__init__(
            inputs=(predictions, target),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        timestamp: int,
    ) -> InformationCoefficientState:
        return InformationCoefficientState(num_stocks=self._num_stocks)

    @staticmethod
    def compute(
        state: InformationCoefficientState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        predictions, target = inputs
        predictions_produced, target_produced = produced

        # Accumulate the cross-sectional correlation on each target tick.
        if target_produced and state.predictions is not None:
            r = target.value()
            s = state.predictions
            valid = np.isfinite(s) & np.isfinite(r)
            s, r = s[valid], r[valid]
            if len(s) >= 2:
                ic = float(np.corrcoef(s, r)[0, 1])
                state.sum_ic += ic
            state.count += 1

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
