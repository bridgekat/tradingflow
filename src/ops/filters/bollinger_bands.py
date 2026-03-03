"""Bollinger Bands filter."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling import Rolling


class BollingerBand[Shape: tuple[int, ...], T: np.floating](Rolling[Shape, T]):
    """Bollinger band at a given standard-deviation offset.

    Outputs ``mean + num_std * std`` over a rolling window, where *num_std*
    is treated as a signed offset (e.g. ``-2`` for the lower band, ``0`` for
    the mean, ``+2`` for the upper band).  Output shape equals the input
    element shape.
    """

    __slots__ = ("_num_std",)

    _num_std: float

    def __init__(
        self,
        window: int | np.timedelta64,
        series: Series[Shape, T],
        num_std: float = 2.0,
    ) -> None:
        super().__init__(window, series)
        self._num_std = num_std

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series[Shape, T]], state: None) -> ArrayLike | None:
        (series,) = inputs
        if not series:
            return None
        vals = self._get_window(series, timestamp)
        if len(vals) <= 1:
            return None
        mean = vals.mean(axis=0)
        std = vals.std(axis=0, ddof=1)
        return mean + self._num_std * std
