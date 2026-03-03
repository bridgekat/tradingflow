"""TradingFlow core package.

This top-level package exposes the core abstractions used across the library:

Type aliases
------------
AnyShape
    ``tuple[int, ...]``; shape of one series element.
Array[Shape, T]
    ``np.ndarray[Shape, np.dtype[T]]``.

Core classes
------------
Series[Shape, T]
    NumPy-backed time series with strictly increasing timestamps.
Operator[Inputs, Shape, T, State]
    Base class for derived-series computations.
Event
    Timestamped mapping of source-series updates.
Scenario
    Event-driven runtime that dispatches updates through an acyclic
    source/operator dependency graph.
"""

from .event import Event
from .operator import Operator
from .scenario import Scenario
from .series import AnyShape, Array, Series

__all__ = [
    "AnyShape",
    "Array",
    "Event",
    "Operator",
    "Scenario",
    "Series",
]
