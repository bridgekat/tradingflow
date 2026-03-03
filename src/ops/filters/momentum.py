"""Momentum filter."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class Momentum[Shape: tuple[int, ...], T: np.floating](Operator[tuple[Series[Shape, T]], Shape, T, None]):
    """Difference between the latest value and value *period* steps ago."""

    __slots__ = ("_period",)

    _period: int

    def __init__(self, period: int, series: Series[Shape, T]) -> None:
        super().__init__((series,), series.shape, series.dtype, None)
        self._period = period

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series[Shape, T]], state: None) -> ArrayLike | None:
        (series,) = inputs
        if len(series) <= self._period:
            return None
        return series.values[-1] - series.values[-self._period - 1]
