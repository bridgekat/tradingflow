"""Weighted moving average filter."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling import Rolling


class WeightedMovingAverage[Shape: tuple[int, ...], T: np.floating](Rolling[Shape, T]):
    """Linearly weighted moving average (WMA).

    Only supports count-based (``int``) windows.  Weights increase linearly
    so the most recent value has the highest weight.
    """

    __slots__ = ()

    def __init__(self, window: int, series: Series[Shape, T]) -> None:
        if not isinstance(window, int):
            raise TypeError(f"WeightedMovingAverage requires an integer window, got {type(window).__name__}")
        super().__init__(window, series)

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series[Shape, T]], state: None) -> tuple[ArrayLike | None, None]:
        (series,) = inputs
        if not series:
            return None, None
        vals = self._get_window(series, timestamp)
        n = len(vals)
        if n == 0:
            return None, None
        weights = np.arange(1, n + 1, dtype=np.float64)
        return np.average(vals, axis=0, weights=weights), None
