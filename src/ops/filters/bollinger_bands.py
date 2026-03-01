"""Bollinger Bands filter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling import Rolling


class BollingerBands(Rolling):
    """Upper, middle, and lower bands around a moving average.

    Produces ``float64`` output regardless of input dtype.
    """

    __slots__ = ("_num_std",)

    def __init__(
        self,
        window: int | np.timedelta64,
        series: Series[Any],
        num_std: float = 2.0,
    ) -> None:
        output_shape = (3, *series.shape)
        super().__init__(window, [series], output_shape)
        self._num_std = num_std

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        series = inputs[0]
        if not series:
            return None
        vals = self._get_window(series, timestamp)
        if len(vals) < 2:
            return None
        mean = vals.mean(axis=0)
        std = vals.std(axis=0, ddof=1)
        lower = mean - self._num_std * std
        upper = mean + self._num_std * std
        return np.stack([lower, mean, upper], axis=0)
