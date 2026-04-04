"""Built-in source implementations for feeding data into the computation graph.

Sources produce timestamped values that drive the event loop. They fall into two
categories:

- **Native sources** -- [`NativeSource`][tradingflow.source.NativeSource] subclasses
  dispatched entirely to Rust (no Python async overhead).
- **Python sources** -- [`Source`][tradingflow.source.Source] subclasses whose
  `init` method returns Python async iterators.

## Historical data sources

- [`ArraySource`][tradingflow.sources.ArraySource] -- backed by in-memory
  `(timestamps, values)` NumPy arrays. Dispatched to Rust for maximum throughput.
  Timestamps must be in non-decreasing order.
- [`CSVSource`][tradingflow.sources.CSVSource] -- backed by a CSV file, parsed and
  ingested in Rust. Requires a [`Schema`][tradingflow.Schema] specifying value
  columns and a timestamp column name.
- [`IterSource`][tradingflow.sources.IterSource] -- driven by a Python iterable of
  `(timestamp, value)` pairs. More flexible than `ArraySource` for lazy or computed
  sequences, but runs in Python. The iterable is materialised at construction time
  for replayability.

## Clock sources (scheduling triggers)

- [`Clock`][tradingflow.sources.Clock] -- fires at explicit timestamps.
- [`DailyClock`][tradingflow.sources.DailyClock] -- fires at midnight each day in a
  given IANA timezone over a `[start, end]` range.
- [`MonthlyClock`][tradingflow.sources.MonthlyClock] -- fires on the first day of
  each month in a given IANA timezone over a `[start, end]` range.

## Sub-modules

- [`stocks`][tradingflow.sources.stocks] -- stock-specific data sources.
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
