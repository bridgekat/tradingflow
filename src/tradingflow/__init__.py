"""TradingFlow core package.

This top-level package exposes the core abstractions used across the library:

Type aliases
------------
AnyShape
    `tuple[int, ...]`; shape of one series element.
Array[Shape, T]
    `np.ndarray[Shape, np.dtype[T]]`; shorthand for array types.

Core classes
------------
Observable[Shape, T]
    Latest-value container.  Every source and operator produces an observable.
Series[Shape, T]
    NumPy-backed time series with strictly increasing timestamps.
    Created by materializing an observable via `Scenario.materialize()`.
Source[Shape, T]
    Abstract base for sources that feed data into an observable via a
    `(historical, live)` async-iterator pair returned by `subscribe()`.
Operator[Inputs, Shape, T, State]
    Base class for derived computations on observables and series.
NativeOperator
    Operator subclass backed by the Rust native extension.  Used by the
    arithmetic factories (``add``, ``negate``, etc.).
Scenario
    Async runtime that subscribes to source streams, accumulates events in the
    point-of-coherency queue (POCQ), and updates an acyclic source/operator
    dependency graph incrementally.
"""

from .observable import Observable
from .series import AnyShape, Array, Series
from .source import Source
from .operator import NativeOperator, Operator
from .scenario import Scenario

__all__ = [
    "AnyShape",
    "Array",
    "NativeOperator",
    "Observable",
    "Operator",
    "Scenario",
    "Series",
    "Source",
]
