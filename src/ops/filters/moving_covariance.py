"""Moving covariance filter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling import Rolling


class MovingCovariance(Rolling):
    """Rolling sample covariance matrix for vector-valued series.

    Produces ``float64`` output regardless of input dtype.
    """

    __slots__ = ("_ddof",)

    def __init__(
        self,
        window: int | np.timedelta64,
        series: Series[Any],
        ddof: int = 1,
    ) -> None:
        if len(series.shape) != 1:
            raise ValueError(
                "MovingCovariance requires a vector-valued series " f"(element shape (n,)), got shape {series.shape}"
            )
        n = series.shape[0]
        super().__init__(window, [series], (n, n))
        self._ddof = ddof

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        series = inputs[0]
        if not series:
            return None
        vals = self._get_window(series, timestamp)
        if len(vals) <= self._ddof:
            return None
        return np.cov(vals.T, ddof=self._ddof)
