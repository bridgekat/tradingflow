"""Built-in sources — the entry points for data into the computation graph.

See the root [`tradingflow`][tradingflow] page for the conceptual
overview (what a source is, historical vs. live streams) and
[`Source`][tradingflow.source.Source] /
[`NativeSource`][tradingflow.source.NativeSource] for the two
implementation tiers.

## Historical data sources

For feeding in pre-recorded data.  Timestamps must be non-decreasing.

- [`ArraySource`][tradingflow.sources.array_source.ArraySource] — backed by a pair
  of in-memory `(timestamps, values)` NumPy arrays.  The fastest
  option when your data is already in memory; dispatched entirely to
  Rust.
- [`CSVSource`][tradingflow.sources.csv_source.CSVSource] — reads rows from a CSV
  file, parsed and ingested in Rust.  Requires a
  [`Schema`][tradingflow.utils.schema.Schema] that names the value columns and the
  timestamp column.  Handles both UTC-labeled and TAI-labeled CSVs via
  the `is_utc` / `tz_offset` parameters — see the timestamp section in
  [`tradingflow`][tradingflow] for background.
- [`IterSource`][tradingflow.sources.iter_source.IterSource] — driven by any
  Python iterable of `(timestamp, value)` pairs.  Less performant than
  `ArraySource` (runs under the GIL), but more flexible for lazy or
  computed sequences.  The iterable is materialized at construction
  time so replays are reproducible.

## Clock sources (scheduling triggers)

Clocks are *unit-valued* sources — they carry no payload, just a
timestamp.  Use them to trigger periodic computations like monthly
rebalancing or daily metric snapshots, either as an explicit input to
metric-style operators or through the
[`Clocked`][tradingflow.operators.clocked.Clocked] wrapper.

- [`Clock`][tradingflow.sources.clock.Clock] — fires at a user-specified
  list of explicit timestamps.  Useful for irregular schedules (e.g.
  rebalance dates read from a file).
- [`DailyClock`][tradingflow.sources.clock.DailyClock] — fires at midnight
  each day in a given IANA timezone, over a `[start, end]` range.
- [`MonthlyClock`][tradingflow.sources.clock.MonthlyClock] — fires on the
  first day of each month in a given IANA timezone, over a
  `[start, end]` range.

## Sub-modules

- [`stocks`][tradingflow.sources.stocks] — sources specific to stock
  data (e.g. financial report CSVs with separate report and notice
  dates).
"""

from . import stocks

from .array_source import ArraySource
from .clock import Clock, DailyClock, MonthlyClock
from .csv_source import CSVSource
from .iter_source import IterSource

__all__ = [
    "stocks",
    "ArraySource",
    "CSVSource",
    "Clock",
    "DailyClock",
    "IterSource",
    "MonthlyClock",
]
