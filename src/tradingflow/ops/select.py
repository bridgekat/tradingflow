"""Vector field-selection operator."""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..operator import Operator
from ..series import Series


class Select[T: np.generic](Operator[tuple[Series[tuple[int], T]], tuple[int], T, None]):
    """Selects a subset of indices from a vector-valued series."""

    __slots__ = ("_indices",)

    _indices: tuple[int, ...]

    def __init__(self, series: Series[tuple[int], T], indices: tuple[int, ...]) -> None:
        if not indices:
            raise ValueError("indices must not be empty.")
        size = series.shape[0]
        for i in indices:
            if not 0 <= i < size:
                raise ValueError(f"index {i} is out of bounds for vector size {size}.")
        super().__init__((series,), (len(indices),), series.dtype)
        self._indices = indices

    @override
    def init_state(self) -> None:
        return None

    @override
    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series[tuple[int], T]],
        state: None,
    ) -> tuple[ArrayLike | None, None]:
        (series,) = inputs
        if not series:
            return None, None
        latest = series[-1]
        return latest[list(self._indices)], None


def select[T: np.generic](series: Series[tuple[int], T], indices: tuple[int, ...]) -> Select[T]:
    """Returns a :class:`Select` operator."""
    return Select(series, indices)
