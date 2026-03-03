"""Moving variance filter."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling import Rolling


class MovingVariance[Shape: tuple[int, ...], T: np.floating](Rolling[Shape, T]):
    """Rolling sample variance (``ddof=1`` by default).

    Output shape matches the input element shape.  Requires more than
    *ddof* observations in the window to produce output.
    """

    __slots__ = ("_ddof",)

    _ddof: int

    def __init__(
        self,
        window: int | np.timedelta64,
        series: Series[Shape, T],
        ddof: int = 1,
    ) -> None:
        super().__init__(window, series)
        self._ddof = ddof

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series[Shape, T]], state: None) -> ArrayLike | None:
        (series,) = inputs
        if not series:
            return None
        vals = self._get_window(series, timestamp)
        if len(vals) <= self._ddof:
            return None
        return vals.var(axis=0, ddof=self._ddof)
