"""Stacking operator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..operator import Operator
from ..series import AnyShape, Series


class Stack[T: np.generic](Operator[tuple[Series[Any, T], ...], AnyShape, T, None]):
    """Stacks N series along a new axis.

    Mirrors :func:`numpy.stack`: all inputs must have the same shape,
    and a new axis of size N is inserted at position *axis*.  Inputs
    without data at or before the timestamp contribute ``NaN``.

    Parameters
    ----------
    inputs
        Input series to stack.  All must have the same shape and dtype.
    axis
        Position of the new axis in the output.  Defaults to ``0``.

    Examples
    --------
    Assemble N per-stock scalar series into a cross-sectional ``(N,)``
    vector::

        series_per_stock = [scenario.add_source(src) for src in stock_sources]
        panel = scenario.add_operator(Stack(series_per_stock))
        # panel shape: (N,)

    Stack N ``(K,)`` vectors into an ``(N, K)`` matrix::

        matrix = scenario.add_operator(Stack(vector_series_list))
    """

    __slots__ = ("_axis",)

    def __init__(self, inputs: list[Series[Any, T]], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("Stack requires at least one input series.")

        self._axis = axis
        dtype = inputs[0].dtype
        element_shape = inputs[0].shape
        out_ndim = len(element_shape) + 1

        if not 0 <= axis < out_ndim:
            raise ValueError(f"axis {axis} is out of bounds for output with {out_ndim} dimensions.")

        for i, inp in enumerate(inputs):
            if inp.shape != element_shape:
                raise ValueError(
                    f"All inputs must have the same shape; "
                    f"input 0 has shape {element_shape}, input {i} has shape {inp.shape}."
                )

        out_shape = list(element_shape)
        out_shape.insert(axis, len(inputs))
        super().__init__(tuple(inputs), tuple(out_shape), dtype)

    def init_state(self) -> None:
        return None

    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series[Any, T], ...],
        state: None,
    ) -> tuple[ArrayLike | None, None]:
        parts: list[np.ndarray] = []

        for series in inputs:
            if not series:
                parts.append(np.full(series.shape, np.nan, dtype=self.dtype))
            else:
                parts.append(series.last)

        return np.stack(parts, axis=self._axis), None
