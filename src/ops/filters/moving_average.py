"""Moving average filter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling import Rolling


class MovingAverage(Rolling):
    """Simple moving average (SMA).

    Produces ``float64`` output regardless of input dtype.
    """

    __slots__ = ()

    def __init__(self, window: int | np.timedelta64, series: Series[Any]) -> None:
        super().__init__(window, [series], series.shape)

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        series = inputs[0]
        if not series:
            return None
        vals = self._get_window(series, timestamp)
        if len(vals) == 0:
            return None
        return vals.mean(axis=0)
