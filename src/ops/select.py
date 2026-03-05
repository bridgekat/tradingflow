"""Vector field-selection operator."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..operator import Operator
from ..series import Series


class Select[T: np.generic](Operator[tuple[Series[tuple[int], T]], tuple[int], T, tuple[int, ...]]):
    """Selects a subset of indices from a vector-valued series."""

    __slots__ = ()

    def __init__(self, series: Series[tuple[int], T], indices: tuple[int, ...]) -> None:
        if not indices:
            raise ValueError("indices must not be empty.")
        size = series.shape[0]
        for i in indices:
            if not 0 <= i < size:
                raise ValueError(f"index {i} is out of bounds for vector size {size}.")
        super().__init__((series,), (len(indices),), series.dtype, indices)

    @override
    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series[tuple[int], T]],
        state: tuple[int, ...],
    ) -> ArrayLike | None:
        (series,) = inputs
        if not series:
            return None
        latest = series[-1]
        return latest[list(state)]


def select[T: np.generic](series: Series[tuple[int], T], indices: tuple[int, ...]) -> Select[T]:
    """Returns a :class:`Select` operator."""
    return Select(series, indices)
