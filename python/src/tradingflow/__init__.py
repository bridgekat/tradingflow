"""TradingFlow — Rust-native computation graph with Python bindings.

TradingFlow is a lightweight library for quantitative investment research
that supports multi-frequency market data, formulaic factors, forecasting
models, portfolio optimization methods and backtesting in a unified data
model.  The core runtime is implemented in Rust; this Python wrapper
provides typed handles, numpy-backed views, and an ergonomic registration
API on top.

# Core concepts

## Arrays and series

A **multi-dimensional array** contains uniformly-typed scalars, backed
by a flat `Vec<T>` scalar buffer that guarantees contiguous layout.

A **time series** contains uniformly-shaped array elements, backed by a
flat `Vec<T>` scalar buffer and a parallel `Vec<Instant>` of
non-decreasing timestamps.  `Instant` is a `#[repr(transparent)]` `i64`
counting SI nanoseconds since the PTP epoch (1970-01-01 00:00:00 TAI).

Most nodes in the computation graph hold either arrays or series as
their output values.  They can be converted back and forth by a pair of
inverse operators, assuming series elements are pushed one-by-one:

* [`Record`][tradingflow.operators.Record] — records every historical
  value of its input node (array) into its output node (series).
* [`Last`][tradingflow.operators.Last] — keeps only the most recent
  value of its input node (series) in its output node (array).

## Sources

A **source** feeds data into a node via asynchronous channels.  A
source implements an `init()` method that returns two channel
receivers: one for historical `(timestamp, event)` tuples and one for
real-time events.  Together they cover the same stream, split at some
instant during the execution of `init()`.

Sources are typically raw market data or pre-computed factors.
Examples include:

* **Order flows** (transactions containing instrument name, price,
  quantity, etc.).
* **Snapshot prices** (arrays of all instrument prices updated at the
  end of each trading day).
* **Financial reports** (arrays of balance sheet fields updated when a
  new financial report is released).

## Operators

An **operator** reads from one or more input nodes and writes data into
an output node.  Its `compute()` method receives the current state,
references to input data, a mutable reference to the output data, and a
flat `produced: tuple[bool, ...]` parallel to the input tuple; it
returns `True` iff downstream nodes should be notified.  `compute()` is
called every time an upstream node notifies.

Operators are the reusable building blocks of trading strategies.
Built-in examples include:

* **Technical indicators** (e.g. 20-day moving average of instrument
  prices).
* **Model predictions** from regression models predicting future
  instrument returns, periodically retrained on historical data
  ([`tradingflow.operators.predictors`][tradingflow.operators.predictors]).
* **Target positions** periodically recomputed by e.g. mean-variance
  portfolio optimisation on forecasted returns and covariances
  ([`tradingflow.operators.portfolios`][tradingflow.operators.portfolios]).
* **Trading simulators** (simulated execution of desired target
  positions with transaction costs, slippage, and other frictions)
  ([`tradingflow.operators.traders`][tradingflow.operators.traders]).
* **Performance metrics** (Sharpe ratios, drawdowns, etc.) calculated
  from past portfolio values
  ([`tradingflow.operators.metrics`][tradingflow.operators.metrics]).

## Scenarios

A [`Scenario`][tradingflow.Scenario] stores and runs the acyclic
computation graph; each node is associated with either a source or an
operator.  [`Scenario.run`][tradingflow.Scenario.run] consumes all
source streams (historical and real-time), puts them in a queue,
coalesces events at the same timestamp, and for each batch propagates
updates through the graph in topological order, ensuring update
timestamps are strictly increasing each batch.

## Timestamp convention

TradingFlow uses **TAI throughout**, on both sides of the PyO3 bridge.
Timestamps are `int64` SI nanoseconds since the PTP epoch 1970-01-01
00:00:00 TAI (`Instant` on the Rust side, reinterpreted as
`datetime64[ns]` on the Python side).  Arithmetic matches NumPy's naïve
`datetime64` semantics exactly: every calendar day is 86 400 SI
seconds, `b - a` yields true elapsed SI time.  No conversion happens at
the FFI edge — the wire format *is* the storage format.

A string like `"2024-01-01"` parsed by NumPy labels the instant
2024-01-01 00:00:00 TAI, which is 2023-12-31 23:59:23 UTC — 37 s earlier
than the same string would mean under a UTC interpretation.  For almost
every backtest this uniform offset is invisible.  When data must be
anchored to wall-clock UTC (ingesting from leap-second-aware systems,
or labelling plot axes), use the conversion helpers in
[`tradingflow.data.time`][tradingflow.data.time]:

* [`utc_to_tai`][tradingflow.data.time.utc_to_tai] — accepts a scalar or
  numpy array.  Adds the current TAI-UTC offset.
* [`tai_to_utc`][tradingflow.data.time.tai_to_utc] — inverse.

String-parsing sources ([`CSVSource`][tradingflow.sources.CSVSource],
[`FinancialReportSource`][tradingflow.sources.stocks.FinancialReportSource])
accept an `is_utc` flag (default `True`) and `tz_offset:
np.timedelta64` so that CSV dates under either interpretation can be
ingested at the source boundary without post-hoc scalar conversion.

## Notification semantics

When a source emits or an operator produces output, its downstream
operators are scheduled.  Each downstream operator receives its flat
`produced` tuple reporting *which* of its inputs produced new output in
the current flush cycle.  This enables two complementary semantics.

**Time-series semantics.**  Each node's output array always holds a
value (the most recent one written).  An operator reads
`inputs[i].value()` to get the latest snapshot, regardless of whether
input `i` produced this cycle.  This is the natural model for state
that persists over time: prices, positions, factor values.  Most
operators use time-series semantics and ignore `produced` entirely,
simply recomputing from the latest values whenever triggered.

**Message-passing semantics.**  A notification is treated as a
*message* whose payload is the value written into the corresponding
node.  If an input did not produce this cycle, there is no message from
it.  Operators inspect `produced[i]` to distinguish "input updated"
from "input is stale".  Message-passing semantics are useful when an
operator must react differently depending on *which* inputs changed —
for example, a forward-adjustment operator that only updates its
cumulative dividend factor when the dividend input produces, and only
emits an adjusted price when the price input produces.  The
[`StackSync`][tradingflow.operators.StackSync] and
[`ConcatSync`][tradingflow.operators.ConcatSync] operators are
message-passing variants of
[`Stack`][tradingflow.operators.Stack] and
[`Concat`][tradingflow.operators.Concat] that fill non-produced input
slots with `NaN`.

**Clocks as first-class inputs.**  A
[`Clock`][tradingflow.sources.Clock] source emits unit events at a
fixed cadence.  Operators that should fire on a schedule rather than on
every data update declare the clock as an explicit unit input in their
input tree — composed naturally with other data inputs.  Two patterns
are common: a **built-in clock input**, as used by the performance
metric operators that take `(data, clock)` and gate their compute
bodies on the clock's produce bit directly; and **external clock
wrapping** via the [`Clocked`][tradingflow.operators.Clocked]
transformer, which prepends a clock input to any operator's inputs and
runs the inner operator only when the clock ticks.  Operators whose
compute depends on message-passing semantics should not be wrapped this
way, since they would miss messages between clock ticks.

# API reference

## Runtime

* [`Scenario`][tradingflow.Scenario] — DAG runtime.  Register sources
  and operators via `add_source` / `add_operator`, then call `run()`.

## Sources and operators

* [`Source`][tradingflow.Source] — abstract base for Python data sources.
* [`NativeSource`][tradingflow.NativeSource] — descriptor for Rust data sources.
* [`Operator`][tradingflow.Operator] — abstract base for Python operators.
* [`NativeOperator`][tradingflow.NativeOperator] — descriptor for Rust operators.
* [`sources`][tradingflow.sources] — built-in source classes.  See the
  sub-module docstring for the full list.
* [`operators`][tradingflow.operators] — built-in operator classes:
  arithmetic, math, structural, series, rolling-window, and Python-side
  operators.  See the sub-module docstring for the full list.

## Handles and type markers

* [`Handle`][tradingflow.Handle] — typed reference to a graph node.
* [`Array`][tradingflow.Array] — marker for Rust `Array<T>` node value types.
* [`Series`][tradingflow.Series] — marker for Rust `Series<T>` node value types.

## Views

* [`ArrayView`][tradingflow.ArrayView] — view of a Rust `Array<T>` node,
  used from inside Python operators to read and write node data.
* [`SeriesView`][tradingflow.SeriesView] — view of a Rust `Series<T>`
  node.

## Utilities

* [`Schema`][tradingflow.Schema] — bidirectional name ↔ position mapping
  for labelling array axes.
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
