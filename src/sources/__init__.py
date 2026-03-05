"""Source adapters for ingesting raw data into source series.

This package defines the source-side ingestion API used by
:class:`src.scenario.Scenario`.

Public API
----------
Source[Shape, T]
    Abstract base class from :mod:`src.source` for one data source bound to
    one source series.
SourceItem[Shape, T]
    Streamed source item type from :mod:`src.source` carrying one value and
    optional payload timestamp.
TimestampMode
    Source timestamp semantics from :mod:`src.source`, either ``"payload"``
    or ``"ingest"``.
CSVSource
    Payload-timestamp adapter for CSV files with configurable column mapping.
ArrayBundleSource
    Payload-timestamp adapter for ``(timestamps, values)`` array bundles.
AsyncCallableSource
    Ingest-timestamp adapter wrapping user-provided async iterables.
eastmoney
    Namespace package grouping EastMoney-specific source adapters under
    :mod:`src.sources.eastmoney.history`.

Invariants
----------
* One source owns exactly one source series.
* Source-level timestamp monotonicity is enforced strictly.
* Values are validated against the bound series shape and dtype before append.
"""

from .array_bundle_source import ArrayBundleSource
from .async_callable_source import AsyncCallableSource
from ..source import Source, SourceItem, TimestampMode
from .csv_source import CSVSource
from . import eastmoney

__all__ = [
    "ArrayBundleSource",
    "AsyncCallableSource",
    "CSVSource",
    "Source",
    "SourceItem",
    "TimestampMode",
    "eastmoney",
]
