"""Portfolio turnover metric."""

from dataclasses import dataclass, field

import numpy as np

from ...views import ArrayView
from ...operator import Operator, Notify
from ...types import Array, Handle, NodeKind


@dataclass(slots=True)
class TurnoverState:
    num_stocks: int
    prev: np.ndarray = field(default_factory=lambda: np.empty(0))
    initialized: bool = False


class Turnover(
    Operator[
        tuple[Handle[Array[np.float64]]],
        Handle[Array[np.float64]],
        TurnoverState,
    ]
):
    """Per-rebalance portfolio turnover.

    On every update of the input weight vector, emits the L1 norm of
    the change since the previous update:

        turnover_t = sum_i |w_{t,i} - w_{t-1,i}|

    For long-only portfolios that sum to 1, turnover lies in ``[0, 2]``:
    ``0`` means no change, ``2`` means a complete liquidation and
    re-investment.

    The first update is a warmup: the operator caches the weights and
    emits ``NaN``.  All subsequent updates emit a finite turnover value.

    NaN handling: positions ``w_{t,i}`` and ``w_{t-1,i}`` that are
    non-finite are treated as ``0`` before computing the difference,
    so a stock going from active to missing (or vice versa) contributes
    its full weight to the turnover.

    Output is a scalar. ``Record(output)`` produces a plottable time
    series; downstream ``RollingMean`` yields an average turnover.

    Parameters
    ----------
    weights
        Soft position weights, shape ``(N,)``.  Typically the output
        of a portfolio constructor.
    """

    def __init__(self, weights: Handle) -> None:
        assert len(weights.shape) == 1
        self._num_stocks = weights.shape[0]

        super().__init__(
            inputs=(weights,),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> TurnoverState:
        return TurnoverState(num_stocks=self._num_stocks)

    @staticmethod
    def compute(
        state: TurnoverState,
        inputs: tuple[ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        notify: Notify,
    ) -> bool:
        current = np.where(np.isfinite(inputs[0].value()), inputs[0].value(), 0.0)

        if not state.initialized:
            state.prev = current
            state.initialized = True
            return False

        turnover = float(np.sum(np.abs(current - state.prev)))
        state.prev = current
        output.write(np.array(turnover, dtype=np.float64))
        return True
