"""Moving covariance filter.

Provides :class:`MovingCovariance`, a rolling sample covariance matrix
operator for vector-valued series.  Input shape ``(n,)`` produces output
shape ``(n, n)``.  Extends :class:`Operator` directly because the output
shape differs from the input element shape.
"""

from __future__ import annotations
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class MovingCovariance[T: np.floating](Operator[tuple[Series[tuple[int], T]], tuple[int, int], T, None]):
    """Rolling sample covariance matrix for vector-valued series.

    Unlike other rolling filters, the output shape ``(n, n)`` differs from
    the input element shape ``(n,)``, so this class extends :class:`Operator`
    directly rather than :class:`Rolling`.
    """

    __slots__ = ("_window", "_ddof")

    _window: int | np.timedelta64
    _ddof: int

    def __init__(
        self,
        window: int | np.timedelta64,
        series: Series[tuple[int], T],
        ddof: int = 1,
    ) -> None:
        if len(series.shape) != 1:
            raise ValueError(
                "MovingCovariance requires a vector-valued series " f"(element shape (n,)), got shape {series.shape}"
            )
        n = series.shape[0]
        super().__init__((series,), (n, n), series.dtype, None)
        self._window = window
        self._ddof = ddof
        if isinstance(window, int):
            if window <= 0:
                raise ValueError("Window size must be a positive integer.")
        else:
            if window <= np.timedelta64(0):
                raise ValueError("Window size must be a positive time delta.")

    def _get_window(self, series: Series[tuple[int], T], timestamp: np.datetime64) -> np.ndarray:
        """Extract values within the rolling window from *series*."""
        if isinstance(self._window, int):
            return series.values[-self._window :]
        else:
            start = timestamp - self._window
            return series.between(start, timestamp, left_inclusive=False, right_inclusive=True).values

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series[tuple[int], T]], state: None) -> ArrayLike | None:
        (series,) = inputs
        if not series:
            return None
        vals = self._get_window(series, timestamp)
        if len(vals) <= self._ddof:
            return None
        return np.cov(vals.T, ddof=self._ddof)
