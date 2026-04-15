"""CSV file source dispatched to Rust for parsing and ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import Schema
from ..source import NativeSource
from ..utils import coerce_timestamp


class CSVSource(NativeSource):
    """Historical source backed by a CSV file, parsed in Rust.

    The CSV must have a date/datetime column and one or more numeric
    value columns. Parsing is handled entirely by the Rust backend.

    Parameters
    ----------
    path
        Path to the CSV file.
    schema
        Column names to load as values (determines element shape and order).
    time_column
        Name of the timestamp column.
    timestamp_offset
        Constant offset added to every parsed timestamp before it is used
        as the event timestamp.  Useful when the CSV contains low-precision
        timestamps (e.g. dates at midnight) that would otherwise cause
        forward-looking bias against higher-precision sources.  Defaults
        to zero (no offset).
    is_utc
        If `True` (default), date strings are interpreted as UTC
        wall-clock instants and converted to this crate's TAI timeline
        via the IERS leap-second table.  If `False`, date strings are
        treated as TAI wall-clock directly (no leap-second math).
    tz_offset
        Offset of the date-string timezone from the reference timescale.
        E.g. `np.timedelta64(8, "h")` for Asia/Shanghai when `is_utc` is
        `True`.  Defaults to zero.
    start
        Optional inclusive start bound.  Rows before this timestamp are
        dropped and the reported time range is clamped.
    end
        Optional inclusive end bound.  Rows after this timestamp are
        dropped and the reported time range is clamped.
    name
        Optional source name.
    """

    def __init__(
        self,
        path: str | Path,
        schema: Schema,
        *,
        time_column: str,
        timestamp_offset: np.timedelta64 = np.timedelta64(0, "ns"),
        is_utc: bool = True,
        tz_offset: np.timedelta64 = np.timedelta64(0, "ns"),
        start: np.datetime64 | None = None,
        end: np.datetime64 | None = None,
        name: str | None = None,
    ) -> None:
        stride = len(schema)
        shape = () if stride == 1 else (stride,)

        offset_ns = int(np.timedelta64(timestamp_offset, "ns").astype(np.int64))
        tz_offset_ns = int(np.timedelta64(tz_offset, "ns").astype(np.int64))

        params: dict = {
            "path": str(Path(path).resolve()),
            "time_column": time_column,
            "value_columns": schema.names,
            "timestamp_offset_ns": offset_ns,
            "is_utc": is_utc,
            "tz_offset_ns": tz_offset_ns,
        }
        if start is not None:
            params["start_ns"] = int(coerce_timestamp(start))
        if end is not None:
            params["end_ns"] = int(coerce_timestamp(end))

        super().__init__(
            native_id="csv",
            dtype="float64",
            shape=shape,
            params=params,
            name=name,
        )
