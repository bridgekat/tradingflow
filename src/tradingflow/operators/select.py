"""Field-selection operator."""

from __future__ import annotations

from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ..operator import Operator
from ..series import AnyShape, Series


class Select[T: np.generic](Operator[tuple[Series[Any, T]], AnyShape, T, None]):
    """Selects indices along one axis, optionally dropping that axis.

    Mirrors NumPy indexing conventions:

    * **Single integer index** (`index: int`) – the selected axis is
      removed from the output shape, like `a[:, 2]`.
    * **Tuple of indices** (`index: tuple[int, ...]`) – the selected
      axis is kept with size `len(index)`, like `a[:, [2, 3]]`.

    Parameters
    ----------
    series
        Input series of any dimensionality (>= 1).
    index
        An `int` for single-index selection (drops the axis), or a
        `tuple[int, ...]` for multi-index selection (keeps the axis).
    axis
        Axis along which to select.  Defaults to `-1` (last axis).
        Negative values count from the end.
    """

    __slots__ = ("_index", "_axis")

    _index: int | list[int]
    _axis: int

    def __init__(
        self,
        series: Series[Any, T],
        index: int | tuple[int, ...],
        *,
        axis: int = -1,
    ) -> None:
        ndim = len(series.shape)
        if ndim == 0:
            raise ValueError("Select requires at least 1 dimension.")

        # Resolve negative axis.
        resolved = axis if axis >= 0 else ndim + axis
        if not 0 <= resolved < ndim:
            raise ValueError(f"axis {axis} is out of bounds for {ndim}-dimensional input.")

        size = series.shape[resolved]

        if isinstance(index, int):
            if not 0 <= index < size:
                raise ValueError(f"index {index} is out of bounds for axis {resolved} size {size}.")
            # Drop the axis.
            out_shape = tuple(d for i, d in enumerate(series.shape) if i != resolved)
            self._index = index
        else:
            if not index:
                raise ValueError("indices must not be empty.")
            for idx in index:
                if not 0 <= idx < size:
                    raise ValueError(f"index {idx} is out of bounds for axis {resolved} size {size}.")
            out_shape = tuple(len(index) if i == resolved else d for i, d in enumerate(series.shape))
            self._index = list(index)

        self._axis = resolved
        super().__init__((series,), out_shape, series.dtype)

    @override
    def init_state(self) -> None:
        return None

    @override
    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series[Any, T]],
        state: None,
    ) -> tuple[ArrayLike | None, None]:
        (series,) = inputs
        if not series:
            return None, None
        latest = series[-1]
        idx: list[Any] = [slice(None)] * latest.ndim
        idx[self._axis] = self._index
        return latest[tuple(idx)], None


def select[T: np.generic](
    series: Series[Any, T],
    index: int | tuple[int, ...],
    *,
    axis: int = -1,
) -> Select[T]:
    """Returns a [`Select`][tradingflow.operators.Select] operator."""
    return Select(series, index, axis=axis)
