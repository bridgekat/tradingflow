"""CSV file source — dispatched to Rust for parsing and ingestion.

Example
-------
```python
from pathlib import Path
from tradingflow import Schema
from tradingflow.sources import CSVSource

schema = Schema(["open", "close", "high", "low", "volume", "amount"])
source = CSVSource(Path("000001_daily_price_raw.csv"), schema, time_column="date")
```
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import Schema
from .clock import NativeSource


class CSVSource(NativeSource):
    """Historical source backed by a CSV file, parsed in Rust.

    The CSV must have a date/datetime column and one or more numeric
    value columns.  Dates are parsed as ``YYYY-MM-DD`` (or
    ``YYYY-MM-DD HH:MM:SS``, truncated to date).

    Parameters
    ----------
    path
        Path to the CSV file.
    schema
        Column names to load as values (determines element shape and order).
    time_column
        Name of the timestamp column.
    name
        Optional source name.
    """

    def __init__(
        self,
        path: str | Path,
        schema: Schema,
        *,
        time_column: str = "date",
        name: str | None = None,
    ) -> None:
        stride = len(schema)
        shape = () if stride == 1 else (stride,)

        super().__init__(
            "csv",
            dtype="float64",
            shape=shape,
            params={
                "path": str(Path(path).resolve()),
                "time_column": time_column,
                "value_columns": schema.names,
            },
            name=name,
        )

        self.schema = schema
