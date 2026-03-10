"""Momentum filter."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class Momentum[Shape: tuple[int, ...], T: np.floating](Operator[tuple[Series[Shape, T]], Shape, T, None]):
    """Difference between the latest value and the value *period* steps ago.

    Parameters
    ----------
    period
        Number of steps to look back.
    series
        Input series to operate on.
    """

    __slots__ = ("_period",)

    _period: int

    def __init__(self, period: int, series: Series[Shape, T]) -> None:
        super().__init__((series,), series.shape, series.dtype)
        self._period = period

    def init_state(self) -> None:
        return None

    @override
    def compute(
        self, timestamp: np.datetime64, inputs: tuple[Series[Shape, T]], state: None
    ) -> tuple[ArrayLike | None, None]:
        (series,) = inputs
        if len(series) <= self._period:
            return None, None
        return series[-1] - series[-self._period - 1], None
