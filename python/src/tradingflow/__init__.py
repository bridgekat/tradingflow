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

## Views (for reading/writing graph data inside Python operators)

* [`ArrayView`][tradingflow.ArrayView] — view of a Rust `Array<T>` node.
* [`SeriesView`][tradingflow.SeriesView] — view of a Rust `Series<T>` node.

[`Operator.compute`][tradingflow.Operator.compute] receives a flat
`produced: tuple[bool, ...]` parallel to its flat `inputs` tuple — no
dedicated "notify" class; `produced[i]` is `True` iff input `i` produced
this flush cycle.

## Utilities

* [`Schema`][tradingflow.Schema] — bidirectional name ↔ position mapping for
  labelling array axes.

## Time convention

TradingFlow uses **TAI throughout**, on both sides of the PyO3 bridge.
Timestamps are `int64` SI nanoseconds since the PTP epoch 1970-01-01
00:00:00 TAI (`Instant` on the Rust side, reinterpreted as `datetime64[ns]`
on the Python side).  Arithmetic matches NumPy's naïve `datetime64`
semantics exactly: every calendar day is 86 400 SI seconds, `b - a`
yields true elapsed SI time.  No conversion happens at the FFI edge —
the wire format *is* the storage format.

A string like `"2024-01-01"` parsed by NumPy labels the instant
2024-01-01 00:00:00 TAI, which is 2023-12-31 23:59:23 UTC — 37 s earlier
than the same string would mean under a UTC interpretation.  For almost
every backtest this uniform offset is invisible.  When data must be
anchored to wall-clock UTC (ingesting from leap-second-aware systems,
or labelling plot axes), use the conversion helpers in
[`tradingflow.data.time`][tradingflow.data.time]:

* [`utc_to_tai`][tradingflow.data.time.utc_to_tai] — accepts a scalar
  or numpy array.  Adds the current TAI−UTC offset.
* [`tai_to_utc`][tradingflow.data.time.tai_to_utc] — inverse.

String-parsing sources (`CSVSource`, `FinancialReportSource`) accept an
`is_utc` flag (default `True`) and `tz_offset: np.timedelta64` so that
CSV dates under either interpretation can be ingested at the source
boundary without post-hoc scalar conversion.
"""

from .data import Array, ArrayView, Handle, NodeKind, Series, SeriesView, Unit
from .source import Source, NativeSource
from .operator import Operator, NativeOperator
from .scenario import Scenario
from .utils import Schema


__all__ = [
    "Array",
    "ArrayView",
    "Handle",
    "NativeOperator",
    "NodeKind",
    "Unit",
    "NativeSource",
    "Operator",
    "Scenario",
    "Schema",
    "Series",
    "SeriesView",
    "Source",
]
