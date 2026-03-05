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
Source[Shape, T]
    Data-source abstraction that owns one source series.
eastmoney
    Namespace package for EastMoney-specific source adapters.
Operator[Inputs, Shape, T, State]
    Base class for derived-series computations.
Scenario
    Async runtime that reads source streams and updates an acyclic
    source/operator dependency graph incrementally.
"""

from .operator import Operator
from .scenario import Scenario
from .series import AnyShape, Array, Series
from .source import Source
from .sources import (
    ArrayBundleSource,
    AsyncCallableSource,
    CSVSource,
    eastmoney,
)

__all__ = [
    "AnyShape",
    "Array",
    "ArrayBundleSource",
    "AsyncCallableSource",
    "CSVSource",
    "Operator",
    "Scenario",
    "Series",
    "Source",
    "eastmoney",
]
