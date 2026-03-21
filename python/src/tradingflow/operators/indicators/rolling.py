"""Rolling window base class for filter operators.

Provides [`Rolling`][tradingflow.operators.indicators.Rolling], an abstract [`Operator`][tradingflow.Operator] subclass that
manages a rolling window specification (count or time-based) and exposes
[`_get_window`][tradingflow.operators.indicators.Rolling._get_window] for concrete filters to extract the relevant slice
of input values.
"""

from __future__ import annotations

from abc import ABC

import numpy as np

from ... import Array, Operator, Series


class Rolling[Shape: tuple[int, ...], T: np.generic](Operator[tuple[Series[Shape, T]], Shape, T, None], ABC):
    """Abstract base for rolling window operators.

    Accepts either a count-based window (`int`, last *N* elements) or a
    time-based window (`np.timedelta64`, elements within a time span).
    Subclasses use [`_get_window`][._get_window] inside their [`compute`][.compute]
    implementation to extract the relevant portion of an input series.

    Parameters
    ----------
    window
        Rolling window specification: an `int` selects the last *N*
        elements; a `np.timedelta64` selects elements within that time
        span before the current timestamp.
    series
        Input series to operate on.
    """

    __slots__ = ("_window",)

    _window: int | np.timedelta64

    def __init__(self, window: int | np.timedelta64, series: Series[Shape, T]) -> None:
        super().__init__((series,), series.shape, series.dtype)
        self._window = window
        if isinstance(window, int):
            if window <= 0:
                raise ValueError("Window size must be a positive integer.")
        else:
            if window <= np.timedelta64(0):
                raise ValueError("Window size must be a positive time delta.")

    def init_state(self) -> None:
        return None

    def _get_window(self, series: Series[Shape, T], timestamp: np.datetime64) -> Array[tuple[int, *Shape], T]:
        """Extract values within the rolling window from *series*."""
        if isinstance(self._window, int):
            return series.values[-self._window :]
        else:
            start = timestamp - self._window
            return series.between(start, timestamp, left_inclusive=False, right_inclusive=True).values
