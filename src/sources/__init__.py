"""Source adapters for ingesting raw data into source series.

This package provides concrete :class:`~src.source.Source` implementations
used by :class:`src.scenario.Scenario`.

Public API
----------
Source[Shape, T]
    Abstract base class from :mod:`src.source`.  Subclass and implement
    :meth:`~src.source.Source.subscribe` to define custom data sources.
CSVSource
    Historical source adapter for CSV files with configurable column mapping.
ArrayBundleSource
    Historical source adapter for ``(timestamps, values)`` array bundles.
AsyncCallableSource
    Live source adapter wrapping user-provided async iterables.
eastmoney
    Namespace package grouping EastMoney-specific source adapters under
    :mod:`src.sources.eastmoney.history`.

Invariants
----------
* One source owns exactly one source series.
* Historical timestamps are validated for strict monotonicity at ingest time.
* Live ingest timestamps are validated for non-strict monotonicity (must not
  decrease relative to the last committed timestamp).
* Values are validated against the bound series shape and dtype before append.
"""

from .array_bundle_source import ArrayBundleSource
from .async_callable_source import AsyncCallableSource
from ..source import Source
from .csv_source import CSVSource
from . import eastmoney

__all__ = [
    "ArrayBundleSource",
    "AsyncCallableSource",
    "CSVSource",
    "Source",
    "eastmoney",
]
