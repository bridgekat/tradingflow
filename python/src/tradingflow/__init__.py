"""TradingFlow — Rust-native computation graph with Python bindings.

This package exposes a directed acyclic graph (DAG) runtime backed by a
compiled Rust core. Users build a graph of sources and operators, then
execute it via the POCQ event loop. The Rust core handles memory layout,
type-erased dispatch, and timestamp-ordered event coalescing; this Python
layer provides typed handles, numpy-backed views, and an ergonomic
registration API.

## Runtime

* [`Scenario`][tradingflow.Scenario] — DAG runtime. Register sources and
  operators via `add_source` / `add_operator`, then call `run()`.

## Sources and operators

* [`Source`][tradingflow.Source] — abstract base for Python data sources.
* [`NativeSource`][tradingflow.NativeSource] — descriptor for Rust data sources.
* [`Operator`][tradingflow.Operator] — abstract base for Python operators.
* [`NativeOperator`][tradingflow.NativeOperator] — descriptor for Rust operators.
* [`sources`][tradingflow.sources] — built-in source classes. See the sub-module
  docstring for the full list.
* [`operators`][tradingflow.operators] — built-in operator classes: arithmetic,
  math, structural, series, rolling-window, and Python-side operators (`Filter`,
  `Where`). See the sub-module docstring for the full list.

## Handles and type markers

* [`Handle`][tradingflow.Handle] — typed reference to a graph node.
* [`Array`][tradingflow.Array] — marker for Rust `Array<T>` node value types.
* [`Series`][tradingflow.Series] — marker for Rust `Series<T>` node value types.

## Views and notifications (for reading/writing graph data inside Python operators)

* [`ArrayView`][tradingflow.ArrayView] — view of a Rust `Array<T>` node.
* [`SeriesView`][tradingflow.SeriesView] — view of a Rust `Series<T>` node.
* [`Notify`][tradingflow.Notify] — notification context passed to
  [`Operator.compute`][tradingflow.Operator.compute] indicating which inputs
  produced new output in the current flush cycle.

## Utilities

* [`Schema`][tradingflow.Schema] — bidirectional name ↔ position mapping for
  labelling array axes.
"""

from .views import ArrayView, Notify, SeriesView
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
    "Notify",
    "NativeSource",
    "Operator",
    "Scenario",
    "Schema",
    "Series",
    "SeriesView",
    "Source",
]
