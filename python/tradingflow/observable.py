"""Observable value — stores only the most recent value of a graph node.

An :class:`Observable` holds a single NumPy array that is overwritten on each
update.  It never stores history.  Any observable can be *materialized* into a
:class:`~tradingflow.series.Series` via :meth:`Scenario.materialize`.

Observables are left uninitialised at creation.  Initial values are set during
graph initialisation: sources provide explicit initial values, and operators
derive theirs by running :meth:`~tradingflow.operator.Operator.compute` once
in topological order.  Missing data is represented via sentinel values (e.g.
NaN for floats).  Timestamps are not stored — they are managed externally by
:class:`~tradingflow.scenario.Scenario`.
"""

from __future__ import annotations

import numpy as np

from .series import AnyShape, Array


class Observable[Shape: AnyShape, T: np.generic]:
    """An observable value that stores only the latest value.

    Parameters
    ----------
    shape
        Shape of each value element.  Use ``()`` for scalars.
    dtype
        NumPy dtype for the value buffer.
    """

    __slots__ = ("_value",)

    _value: Array[Shape, T]

    def __init__(
        self,
        shape: Shape,
        dtype: type[T] | np.dtype[T],
    ) -> None:
        self._value: Array[Shape, T] = np.empty(shape, dtype=dtype)  # type: ignore[assignment]

    @property
    def shape(self) -> Shape:
        """Element shape of the stored value."""
        return self._value.shape

    @property
    def dtype(self) -> np.dtype[T]:
        """NumPy dtype of the stored value."""
        return self._value.dtype

    @property
    def last(self) -> Array[Shape, T]:
        """The current value as an ``ndarray``."""
        return self._value

    def write(self, value: Array[Shape, T]) -> None:
        """Overwrite the current value."""
        self._value = value
