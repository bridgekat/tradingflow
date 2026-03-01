"""Average return metric operator."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class AverageReturn(Operator[dict[str, Any], np.float64]):
    """Average period-over-period return, updated on a periodic signal."""

    __slots__ = ()

    def __init__(
        self,
        market_values: Series[Any],
        signal: Series[Any],
    ) -> None:
        state: dict[str, Any] = {
            "prev_value": None,
            "sum_returns": 0.0,
            "count": 0,
        }
        super().__init__([market_values, signal], state, np.dtype(np.float64))

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        mv_series, signal = inputs[0], inputs[1]
        if not mv_series or not signal:
            return None

        last_signal = signal.last
        if last_signal is None or last_signal[0] != timestamp:
            return None

        current = float(mv_series.values[-1])

        if self._state["prev_value"] is not None:
            prev = self._state["prev_value"]
            if prev != 0.0:
                ret = (current - prev) / prev
                self._state["sum_returns"] += ret
                self._state["count"] += 1

        self._state["prev_value"] = current

        if self._state["count"] == 0:
            return None
        return self._state["sum_returns"] / self._state["count"]
