"""Element-wise conditional operator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ..observable import Observable
from ..operator import Operator
from ..series import AnyShape


class Where[Shape: AnyShape, T: np.generic](Operator[tuple[Observable[Shape, T]], Shape, T, None]):
    """Element-wise conditional: keeps values where *fn* is `True`.

    At each timestamp the latest value is tested element-wise with *fn*.
    Elements where *fn* returns `False` are replaced with *fill*
    (default `NaN`).  Equivalent to `np.where(fn(x), x, fill)`.

    Parameters
    ----------
    series
        Input series.
    fn
        Element-wise predicate.  Receives the latest value (an ndarray)
        and must return a boolean array broadcastable to the same shape.
    fill
        Replacement value for elements where *fn* is `False`.
        Defaults to `np.nan`.

    Examples
    --------
    Keep only positive values, NaN elsewhere::

        positive_mcap = Where(mcap_series, fn=lambda x: x > 0)
    """

    __slots__ = ("_fn", "_fill")

    def __init__(
        self,
        series: Observable[Shape, T],
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        fill: float = np.nan,
    ) -> None:
        self._fn = fn
        self._fill = fill
        super().__init__((series,), series.shape, series.dtype)

    @override
    def init_state(self) -> None:
        return None

    @override
    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Observable[Shape, T]],
        state: None,
    ) -> tuple[ArrayLike | None, None]:
        (obs,) = inputs
        latest = obs.last
        mask = self._fn(latest)
        return np.where(mask, latest, self._fill), None
