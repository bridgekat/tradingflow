"""Portfolio turnover metric."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class TurnoverState:
    num_stocks: int
    prev: np.ndarray = field(default_factory=lambda: np.empty(0))
    initialized: bool = False


class Turnover:
    r"""Per-rebalance portfolio turnover.

    On every update of the input weight vector, emits the L1 norm of
    the change since the previous update:
    \(\text{turnover}_t = \sum_i |w_{t,i} - w_{t-1,i}|\).

    For long-only portfolios that sum to 1, turnover lies in `[0, 2]`:
    `0` means no change, `2` means a complete liquidation and
    re-investment.

    The first update is a warmup: the operator caches the weights and
    emits `NaN`.  All subsequent updates emit a finite turnover value.

    NaN handling: positions \(w_{t,i}\) and \(w_{t-1,i}\) that are
    non-finite are treated as `0` before computing the difference,
    so a stock going from active to missing (or vice versa) contributes
    its full weight to the turnover.

    Output is a scalar. `Record(output)` produces a plottable time
    series; downstream `RollingMean` yields an average turnover.

    Parameters
    ----------
    num_stocks
        Length `N` of the soft position weight vector.
    """

    def __init__(self, num_stocks: int) -> None:
        self._num_stocks = num_stocks

    def init(self, inputs, timestamp: int) -> TurnoverState:
        return TurnoverState(num_stocks=self._num_stocks)

    @staticmethod
    def compute(
        state: TurnoverState,
        inputs,
        output,
        timestamp: int,
        produced: tuple[bool, ...],
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


def build(**kwargs) -> Turnover:
    """Construct a :class:`Turnover` operator.

    Build kwargs
    ------------
    num_stocks : int
        Length `N` of the input weight vector.
    """
    return Turnover(num_stocks=int(kwargs["num_stocks"]))
