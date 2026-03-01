"""TradingFlow – a lightweight library for quantitative investment research.

This top-level package provides the core time series primitives.

Classes
-------
Series[T]
    A NumPy-backed time series of ``(timestamp, value)`` pairs with strictly
    increasing ``np.datetime64[ns]`` timestamps.  Timestamps are stored in a
    1-D ``NDArray[np.datetime64]`` and values in an ``NDArray[T]`` where *T*
    is a ``np.generic`` subtype and each element may be N-dimensional.  All
    elements share the same shape and dtype.  Internal buffers are always
    allocated at construction time with pre-determined dtypes
    (``dtype``, ``shape``).  Supports integer indexing, slicing,
    as-of timestamp lookup (:meth:`~Series.at`), time-range queries
    (:meth:`~Series.between`), and as-of slicing (:meth:`~Series.to`).

Operator[S, T]
    Abstract base for operators that compute derived time series.
    Subclass and override :meth:`~Operator.compute` to define custom
    logic.  Each call to :meth:`~Operator.update` slices all inputs up
    to the given timestamp, passes them to :meth:`~Operator.compute`,
    and appends the result to an internal :attr:`~Operator.output`
    :class:`Series` when the result is not ``None``.

See README.md and AGENTS.md for project overview and contributor guidance.
"""

from .operator import Operator
from .series import Series

__all__ = [
    "Operator",
    "Series",
]
