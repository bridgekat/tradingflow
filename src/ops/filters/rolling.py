"""Rolling window base class for filter operators."""

from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ... import Operator, Series


class Rolling(Operator[None, np.float64], ABC):
    """Abstract base for rolling window operators.

    Accepts either a count-based window (``int``, last *N* elements) or a
    time-based window (``np.timedelta64``, elements within a time span).
    Subclasses use :meth:`_get_window` inside their :meth:`compute`
    implementation to extract the relevant portion of an input series.

    All rolling filters produce ``float64`` output regardless of input dtype.
    """

    __slots__ = ("_window",)

    def __init__(
        self,
        window: int | np.timedelta64,
        inputs: list[Series[Any]],
        shape: tuple[int, ...] = (),
    ) -> None:
        super().__init__(inputs, None, np.dtype(np.float64), shape)
        self._window: int | np.timedelta64 = window

    @property
    def window(self) -> int | np.timedelta64:
        """The rolling window size."""
        return self._window

    def _get_window(self, series: Series[Any], timestamp: np.datetime64) -> NDArray[Any]:
        """Extract values within the rolling window from *series*."""
        if isinstance(self._window, np.timedelta64):
            start = timestamp - self._window
            return series.between(start, timestamp).values
        return series.values[-int(self._window) :]
