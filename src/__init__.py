"""TradingFlow – a lightweight library for quantitative investment research.

This top-level package provides the core time series primitives and
re-exports them for convenient access.

Type aliases
------------
AnyShape
    ``tuple[int, ...]`` – element shape of a series value.
Array[Shape, T]
    ``np.ndarray[Shape, np.dtype[T]]`` – shaped NumPy array.

Classes
-------
Series[Shape, T]
    A NumPy-backed time series of ``(timestamp, value)`` pairs with strictly
    increasing ``np.datetime64[ns]`` timestamps.  Scalars are stored as
    0-dimensional arrays (shape ``()``) to keep the API uniform.  Supports
    integer indexing, slicing, as-of timestamp lookup (:meth:`~Series.at`),
    time-range queries (:meth:`~Series.between`), and as-of slicing
    (:meth:`~Series.to`).

Operator[Inputs, Shape, T, State]
    Abstract base for operators that compute derived time series.
    Subclass and override :meth:`~Operator.compute` to define custom
    logic.  Each call to :meth:`~Operator.update` slices all inputs up
    to the given timestamp, passes them to :meth:`~Operator.compute`,
    and appends the result to an internal :attr:`~Operator.output`
    :class:`Series` when the result is not ``None``.

See README.md and AGENTS.md for project overview and contributor guidance.
"""

from .operator import Operator
from .series import AnyShape, Array, Series

__all__ = [
    "AnyShape",
    "Array",
    "Operator",
    "Series",
]
