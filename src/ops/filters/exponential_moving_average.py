"""Exponential moving average filter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class ExponentialMovingAverage(Operator[None, np.float64]):
    """Exponential moving average (EMA).

    Produces ``float64`` output regardless of input dtype.
    """

    __slots__ = ("_alpha",)

    def __init__(self, alpha: float, series: Series[Any]) -> None:
        super().__init__([series], None, np.dtype(np.float64), series.shape)
        self._alpha = alpha

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        series = inputs[0]
        if not series:
            return None
        latest = series.values[-1]
        if self.output:
            prev = self.output.values[-1]
            return self._alpha * latest + (1.0 - self._alpha) * prev
        return latest
