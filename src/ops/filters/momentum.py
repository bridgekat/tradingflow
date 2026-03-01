"""Momentum filter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class Momentum(Operator[None, np.float64]):
    """Difference between the latest value and value *period* steps ago.

    Produces ``float64`` output regardless of input dtype.
    """

    __slots__ = ("_period",)

    def __init__(self, period: int, series: Series[Any]) -> None:
        super().__init__([series], None, np.dtype(np.float64), series.shape)
        self._period = period

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        series = inputs[0]
        if len(series) <= self._period:
            return None
        return series.values[-1] - series.values[-self._period - 1]
