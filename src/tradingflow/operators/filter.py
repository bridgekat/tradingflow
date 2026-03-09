"""Whole-element filter operator."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..operator import Operator
from ..series import AnyShape, Series


class Filter[Shape: AnyShape, T: np.generic](Operator[tuple[Series[Shape, T]], Shape, T, None]):
    """Filters entire elements by a scalar predicate.

    At each timestamp the latest value is passed to *fn*, which must return
    a single boolean.  If ``True`` the value is emitted as-is; if ``False``
    no output is produced for this timestamp (i.e. ``compute`` returns
    ``None``).

    This differs from :class:`~tradingflow.operators.Where`, which replaces
    individual array elements with a fill value.  ``Filter`` drops the
    entire element when the predicate is ``False``.

    Parameters
    ----------
    series
        Input series.
    fn
        Predicate receiving the latest value (an ndarray) and returning a
        single ``bool``.

    Examples
    --------
    Keep only timestamps where the sum is positive::

        filtered = Filter(series, fn=lambda x: float(np.sum(x)) > 0)
    """

    __slots__ = ("_fn",)

    def __init__(
        self,
        series: Series[Shape, T],
        fn: Callable[[np.ndarray], bool],
    ) -> None:
        self._fn = fn
        super().__init__((series,), series.shape, series.dtype)

    @override
    def init_state(self) -> None:
        return None

    @override
    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series[Shape, T]],
        state: None,
    ) -> tuple[ArrayLike | None, None]:
        (series,) = inputs
        if not series:
            return None, None
        latest = series[-1]
        if self._fn(latest):
            return latest, None
        return None, None
