"""TradingFlow — Rust-native computation graph with Python bindings.

Core classes:

* [`Scenario`][tradingflow.Scenario] — async runtime for the DAG.
* [`Source`][tradingflow.Source] — abstract base for Python data sources.
* [`Operator`][tradingflow.Operator] — abstract base for Python operators.
* [`NativeOperator`][tradingflow.NativeOperator] — descriptor for Rust operators.
* [`Handle`][tradingflow.Handle] — typed reference to a graph node.

Type markers (for generic parameters):

* [`Array`][tradingflow.Array] — marker for Rust `Array<T>` nodes.
* [`Series`][tradingflow.Series] — marker for Rust `Series<T>` nodes.

View types (for Python operators reading/writing graph data):

* [`ArrayView`][tradingflow.ArrayView] — view of a Rust `Array<T>` node.
* [`SeriesView`][tradingflow.SeriesView] — view of a Rust `Series<T>` node.
"""

from .operator import NativeOperator, Operator
from .scenario import Scenario
from .source import Source
from .types import Array, Handle, Series
from .views import ArrayView, SeriesView

__all__ = [
    "Array",
    "ArrayView",
    "Handle",
    "NativeOperator",
    "Operator",
    "Scenario",
    "Series",
    "SeriesView",
    "Source",
]
