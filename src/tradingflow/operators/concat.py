"""Concatenation operator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..operator import Operator
from ..series import AnyShape, Series


class Concat[T: np.generic](Operator[tuple[Series[Any, T], ...], AnyShape, T, None]):
    """Concatenates N series along an existing axis.

    Mirrors :func:`numpy.concatenate`: all inputs must have at least one
    dimension and must agree on every axis except *axis*, where sizes are
    summed.  Inputs without data at or before the timestamp contribute
    ``NaN``.

    Parameters
    ----------
    inputs
        Input series to concatenate.  All must have the same dtype, the
        same number of dimensions (>= 1), and matching sizes on every
        axis except *axis*.
    axis
        Existing axis along which to concatenate.  Defaults to ``0``.

    Examples
    --------
    Concatenate two ``(K,)`` vectors into ``(2K,)``::

        combined = scenario.add_operator(Concat([features_a, features_b]))
    """

    __slots__ = ("_axis",)

    def __init__(self, inputs: list[Series[Any, T]], *, axis: int = 0) -> None:
        if not inputs:
            raise ValueError("Concat requires at least one input series.")

        self._axis = axis
        dtype = inputs[0].dtype
        shapes = [inp.shape for inp in inputs]
        ndim = len(shapes[0])

        if not 0 <= axis < ndim:
            raise ValueError(f"axis {axis} is out of bounds for {ndim}-dimensional inputs.")

        for i, sh in enumerate(shapes):
            if len(sh) != ndim:
                raise ValueError(
                    f"All inputs must have the same number of dimensions; "
                    f"input 0 has {ndim}, input {i} has {len(sh)}."
                )
            for d in range(ndim):
                if d != axis and sh[d] != shapes[0][d]:
                    raise ValueError(
                        f"All inputs must match on non-concatenation axes; "
                        f"input 0 has size {shapes[0][d]} on axis {d}, "
                        f"input {i} has size {sh[d]}."
                    )

        out_shape = list(shapes[0])
        out_shape[axis] = sum(sh[axis] for sh in shapes)
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
            val = series.last
            if val is None:
                parts.append(np.full(series.shape, np.nan, dtype=self.dtype))
            else:
                parts.append(val)

        return np.concatenate(parts, axis=self._axis), None
