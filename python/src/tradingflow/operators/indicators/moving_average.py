"""Moving average filter."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling import Rolling


class MovingAverage[Shape: tuple[int, ...], T: np.floating](Rolling[Shape, T]):
    """Simple moving average (SMA)."""

    __slots__ = ()

    def __init__(self, window: int | np.timedelta64, series: Series[Shape, T]) -> None:
        super().__init__(window, series)

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series[Shape, T]], state: None) -> tuple[ArrayLike | None, None]:
        (series,) = inputs
        if not series:
            return None, None
        vals = self._get_window(series, timestamp)
        if len(vals) == 0:
            return None, None
        return vals.mean(axis=0), None
