"""Exponential moving average filter."""

from __future__ import annotations

from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ... import Array, Operator
from ...observable import Observable


class ExponentialMovingAverage[Shape: tuple[int, ...], T: np.floating](
    Operator[tuple[Observable[Shape, T]], Shape, T, Array[Any, Any] | None]
):
    """Exponential moving average (EMA).

    Unlike [`Rolling`][tradingflow.operators.indicators.Rolling]-based filters, EMA extends [`Operator`][tradingflow.Operator]
    directly because it uses exponential weighting rather than a fixed
    window.  The previous EMA value is carried in the operator *state*
    (`None` before the first observation, then an `ndarray`).

    Parameters
    ----------
    alpha
        Smoothing factor in `(0, 1]`.  Larger values weight recent
        observations more heavily.
    series
        Input series to operate on.
    """

    __slots__ = ("_alpha",)

    _alpha: float

    def __init__(self, alpha: float, series: Observable[Shape, T]) -> None:
        super().__init__((series,), series.shape, series.dtype)
        self._alpha = alpha

    @override
    def init_state(self) -> Array[Any, Any] | None:
        return None

    @override
    def compute(
        self, timestamp: np.datetime64, inputs: tuple[Observable[Shape, T]], state: Array[Any, Any] | None
    ) -> tuple[ArrayLike | None, Array[Any, Any] | None]:
        (obs,) = inputs
        latest = obs.last
        if state is not None:
            result = self._alpha * latest + (1.0 - self._alpha) * state
        else:
            result = latest
        return result, result
