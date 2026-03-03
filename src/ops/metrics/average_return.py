"""Average return metric."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class AverageReturn(Operator[tuple[Series, Series], tuple[()], np.float64, dict]):
    """Computes the running average return triggered by a signal series.

    At each timestamp where the signal series has data, a return is computed
    from the change in market value since the previous signal observation.
    The output is the cumulative mean of all observed returns.
    """

    __slots__ = ()

    def __init__(self, market_values: Series, signal: Series) -> None:
        state = {"prev_mv": None, "sum_returns": 0.0, "count": 0}
        super().__init__((market_values, signal), (), np.dtype(np.float64), state)

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series, Series], state: dict) -> ArrayLike | None:
        mv, signal = inputs
        if not signal or not mv:
            return None
        if state["prev_mv"] is None:
            state["prev_mv"] = float(mv.values[-1])
            return None
        current_mv = float(mv.values[-1])
        ret = (current_mv - state["prev_mv"]) / state["prev_mv"]
        state["prev_mv"] = current_mv
        state["sum_returns"] += ret
        state["count"] += 1
        return state["sum_returns"] / state["count"]
