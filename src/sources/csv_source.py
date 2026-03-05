"""CSV-backed payload-timestamp source."""

from __future__ import annotations

import csv
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..series import AnyShape, Series
from ..source import Source, SourceItem

type _RowConverter = Callable[[str], Any]
type _TimestampParser = Callable[[str], np.datetime64]


class CSVSource[Shape: AnyShape, T: np.generic](Source[Shape, T]):
    """Reads one CSV file and emits payload-timestamp source items.

    Parameters
    ----------
    path
        CSV file path.
    series
        Destination source series.
    timestamp_col
        Column containing timestamp payloads.
    value_cols
        Ordered columns forming one emitted value.
    delimiter
        CSV delimiter.
    encoding
        File encoding.
    timestamp_parser
        Optional parser applied to the timestamp column.
    converters
        Optional per-column parsers for value columns.
    name
        Optional source name.
    """

    __slots__ = (
        "_path",
        "_timestamp_col",
        "_value_cols",
        "_delimiter",
        "_encoding",
        "_timestamp_parser",
        "_converters",
    )

    _path: Path
    _timestamp_col: str
    _value_cols: tuple[str, ...]
    _delimiter: str
    _encoding: str
    _timestamp_parser: _TimestampParser | None
    _converters: dict[str, _RowConverter]

    def __init__(
        self,
        path: str | Path,
        series: Series[Shape, T],
        *,
        timestamp_col: str,
        value_cols: Sequence[str],
        delimiter: str = ",",
        encoding: str = "utf-8",
        timestamp_parser: _TimestampParser | None = None,
        converters: Mapping[str, _RowConverter] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(series, name=name, timestamp_mode="payload")
        self._path = Path(path)
        self._timestamp_col = timestamp_col
        self._value_cols = tuple(value_cols)
        self._delimiter = delimiter
        self._encoding = encoding
        self._timestamp_parser = timestamp_parser
        self._converters = dict(converters or {})

        if not self._value_cols:
            raise ValueError("value_cols must not be empty.")
        if self.series.shape == ():
            if len(self._value_cols) != 1:
                raise ValueError("Scalar series require exactly one value column.")
        else:
            expected_columns = int(np.prod(self.series.shape))
            if len(self._value_cols) != expected_columns:
                raise ValueError(
                    f"value_cols count {len(self._value_cols)} does not match "
                    f"series element size {expected_columns} for shape {self.series.shape}"
                )

    async def stream(self) -> AsyncIterator[SourceItem[Shape, T]]:
        with self._path.open("r", encoding=self._encoding, newline="") as file:
            reader = csv.DictReader(file, delimiter=self._delimiter)
            fieldnames = set(reader.fieldnames or ())

            required_columns = {self._timestamp_col, *self._value_cols}
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise ValueError(f"CSV source '{self.name}' is missing required columns: {missing}")

            for row_index, row in enumerate(reader, start=2):
                timestamp = self._parse_timestamp(row, row_index)
                values = self._parse_values(row, row_index)
                yield SourceItem(value=values, timestamp=timestamp)

    def _parse_timestamp(self, row: dict[str, str], row_index: int) -> np.datetime64:
        raw = row[self._timestamp_col]
        try:
            if self._timestamp_parser is not None:
                return self._timestamp_parser(raw)
            return np.datetime64(raw)
        except Exception as exc:
            raise ValueError(
                f"CSV source '{self.name}' could not parse timestamp "
                f"at row {row_index} column '{self._timestamp_col}': {raw!r}"
            ) from exc

    def _parse_values(self, row: dict[str, str], row_index: int) -> np.ndarray[Any, np.dtype[Any]] | Any:
        parsed: list[Any] = []
        for col in self._value_cols:
            raw = row[col]
            converter = self._converters.get(col)
            try:
                parsed.append(converter(raw) if converter is not None else raw)
            except Exception as exc:
                raise ValueError(
                    f"CSV source '{self.name}' converter failed at row {row_index} column '{col}': {raw!r}"
                ) from exc

        if self.series.shape == ():
            return parsed[0]

        flat = np.asarray(parsed)
        return cast(np.ndarray[Any, np.dtype[Any]], flat.reshape(self.series.shape))
