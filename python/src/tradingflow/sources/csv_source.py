"""CSV file source dispatched to Rust for parsing and ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import Schema
from ..source import NativeSource


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
