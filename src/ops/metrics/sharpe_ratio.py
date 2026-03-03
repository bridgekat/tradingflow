"""Sharpe ratio metric."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class SharpeRatio(Operator[tuple[Series, Series], tuple[()], np.float64, dict]):
    """Annualised Sharpe ratio computed on signal-triggered returns.

    Outputs only when at least two returns have been observed and the
    standard deviation is non-zero.
    """

    __slots__ = ("_periods_per_year",)

    _periods_per_year: int

    def __init__(self, market_values: Series, signal: Series, periods_per_year: int = 252) -> None:
        state = {"prev_mv": None, "returns": []}
        super().__init__((market_values, signal), (), np.dtype(np.float64), state)
        self._periods_per_year = periods_per_year

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
        state["returns"].append(ret)
        if len(state["returns"]) < 2:
            return None
        returns = np.array(state["returns"])
        std = returns.std(ddof=1)
        if std == 0:
            return None
        return float(returns.mean() / std * np.sqrt(self._periods_per_year))
