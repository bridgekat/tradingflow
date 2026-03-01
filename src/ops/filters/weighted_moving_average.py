"""Weighted moving average filter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling import Rolling


class WeightedMovingAverage(Rolling):
    """Linearly weighted moving average for count-based windows.

    Produces ``float64`` output regardless of input dtype.
    """

    __slots__ = ()

    def __init__(self, window: int, series: Series[Any]) -> None:
        if isinstance(window, np.timedelta64):
            raise TypeError("WeightedMovingAverage requires an integer window, " f"got {type(window).__name__}")
        super().__init__(int(window), [series], series.shape)

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        series = inputs[0]
        if not series:
            return None
        vals = self._get_window(series, timestamp)
        n = len(vals)
        if n == 0:
            return None
        weights = np.arange(1, n + 1, dtype=np.float64)
        weights /= weights.sum()
        if vals.ndim > 1:
            shape = (n,) + (1,) * (vals.ndim - 1)
            weights = weights.reshape(shape)
        return (vals * weights).sum(axis=0)
