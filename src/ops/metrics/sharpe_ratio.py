"""Sharpe ratio metric operator."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ... import Operator, Series


class SharpeRatio(Operator[dict[str, Any], np.float64]):
    """Annualised Sharpe ratio, updated on a periodic signal."""

    __slots__ = ("_periods_per_year",)

    def __init__(
        self,
        market_values: Series[Any],
        signal: Series[Any],
        periods_per_year: int = 252,
    ) -> None:
        state: dict[str, Any] = {
            "prev_value": None,
            "returns": [],
        }
        super().__init__([market_values, signal], state, np.dtype(np.float64))
        self._periods_per_year = periods_per_year

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
                self._state["returns"].append(ret)

        self._state["prev_value"] = current

        returns = self._state["returns"]
        if len(returns) < 2:
            return None

        arr: NDArray[np.float64] = np.array(returns, dtype=np.float64)
        mean_ret = float(arr.mean())
        std_ret = float(arr.std(ddof=1))
        if std_ret == 0.0:
            return None
        return mean_ret / std_ret * np.sqrt(self._periods_per_year)
