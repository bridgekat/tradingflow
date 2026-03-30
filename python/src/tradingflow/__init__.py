"""TradingFlow — Rust-native computation graph with Python bindings.

Core classes:

* [`Scenario`][tradingflow.Scenario] — async runtime for the DAG.
* [`Source`][tradingflow.Source] — abstract base for Python data sources.
* [`NativeSource`][tradingflow.NativeSource] — descriptor for Rust data sources.
* [`Operator`][tradingflow.Operator] — abstract base for Python operators.
* [`NativeOperator`][tradingflow.NativeOperator] — descriptor for Rust operators.
* [`Handle`][tradingflow.Handle] — typed reference to a graph node.
* [`Schema`][tradingflow.Schema] — bidirectional name↔position mapping for array axes.

Type markers (for generic parameters):

* [`Array`][tradingflow.Array] — marker for Rust `Array<T>` nodes.
* [`Series`][tradingflow.Series] — marker for Rust `Series<T>` nodes.

View types (for Python operators reading/writing graph data):

* [`ArrayView`][tradingflow.ArrayView] — view of a Rust `Array<T>` node.
* [`SeriesView`][tradingflow.SeriesView] — view of a Rust `Series<T>` node.
"""

from .views import ArrayView, SeriesView
from .source import Source, NativeSource
from .operator import Operator, NativeOperator
from .types import Array, Series, Handle
from .scenario import Scenario
from .schema import Schema

__all__ = [
    "Array",
    "ArrayView",
    "Handle",
    "NativeOperator",
    "NativeSource",
    "Operator",
    "Scenario",
    "Schema",
    "Series",
    "SeriesView",
    "Source",
]
