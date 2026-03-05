"""EastMoney history adapter for raw daily market snapshot CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ....series import Series
from ....source import Source, SourceItem


@dataclass(slots=True, frozen=True)
class DailyMarketSnapshotSchema:
    """Canonical schema for daily market snapshot vectors."""

    field_ids: tuple[str, ...]
    field_index: dict[str, int]

    @classmethod
    def from_field_ids(cls, field_ids: tuple[str, ...]) -> DailyMarketSnapshotSchema:
        """Builds a schema from ordered field identifiers."""
        if not field_ids:
            raise ValueError("field_ids must not be empty.")
        if len(set(field_ids)) != len(field_ids):
            raise ValueError("field_ids must be unique.")
        return cls(field_ids=field_ids, field_index={field_id: i for i, field_id in enumerate(field_ids)})


DEFAULT_DAILY_MARKET_SNAPSHOT_SCHEMA = DailyMarketSnapshotSchema.from_field_ids(
    ("open", "close", "high", "low", "amount", "volume")
)


@dataclass(slots=True, frozen=True)
class DailyMarketSnapshotDiagnostics:
    """Diagnostics for daily snapshot parsing."""

    dropped_rows: int
    total_rows: int
    emitted_rows: int

    @staticmethod
    def empty() -> DailyMarketSnapshotDiagnostics:
        """Returns an empty diagnostics record."""
        return DailyMarketSnapshotDiagnostics(dropped_rows=0, total_rows=0, emitted_rows=0)


class DailyMarketSnapshotCSVSource(Source[tuple[int], np.float64]):
    """Payload-timestamp source for raw daily price market snapshots.

    Expected raw columns:
    ``date``, ``open``, ``close``, ``high``, ``low``, ``amount``, ``volume``.

    Output vector order follows :attr:`schema.field_ids`.
    The raw ``volume`` is scaled by ``volume_lot_size`` (default ``100``),
    matching the conventions in the reference preprocessing code.
    """

    __slots__ = (
        "_path",
        "_schema",
        "_volume_lot_size",
        "_strict_row_checks",
        "_diagnostics",
    )

    _path: Path
    _schema: DailyMarketSnapshotSchema
    _volume_lot_size: int
    _strict_row_checks: bool
    _diagnostics: DailyMarketSnapshotDiagnostics

    def __init__(
        self,
        path: str | Path,
        *,
        schema: DailyMarketSnapshotSchema | None = None,
        volume_lot_size: int = 100,
        strict_row_checks: bool = True,
        name: str | None = None,
    ) -> None:
        schema_resolved = schema or DEFAULT_DAILY_MARKET_SNAPSHOT_SCHEMA
        series = Series((len(schema_resolved.field_ids),), np.dtype(np.float64))
        super().__init__(series, name=name, timestamp_mode="payload")
        self._path = Path(path)
        self._schema = schema_resolved
        self._volume_lot_size = volume_lot_size
        self._strict_row_checks = strict_row_checks
        self._diagnostics = DailyMarketSnapshotDiagnostics.empty()
        if self._volume_lot_size <= 0:
            raise ValueError("volume_lot_size must be positive.")

    @property
    def schema(self) -> DailyMarketSnapshotSchema:
        """Canonical schema used by this source."""
        return self._schema

    @property
    def diagnostics(self) -> DailyMarketSnapshotDiagnostics:
        """Latest parsing diagnostics."""
        return self._diagnostics

    async def stream(self):
        required_columns = {"date", "open", "close", "high", "low", "amount", "volume"}

        dropped_rows = 0
        total_rows = 0
        emitted_rows = 0

        with self._path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or ())
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise ValueError(f"Daily market snapshot source '{self.name}' is missing required columns: {missing}")

            for row_index, row in enumerate(reader, start=2):
                total_rows += 1
                try:
                    timestamp, value = self._parse_row(row, row_index)
                except ValueError:
                    dropped_rows += 1
                    if self._strict_row_checks:
                        raise
                    continue
                emitted_rows += 1
                yield SourceItem(value=value, timestamp=timestamp)

        self._diagnostics = DailyMarketSnapshotDiagnostics(
            dropped_rows=dropped_rows,
            total_rows=total_rows,
            emitted_rows=emitted_rows,
        )

    def _parse_row(
        self,
        row: dict[str, str],
        row_index: int,
    ) -> tuple[np.datetime64, np.ndarray[tuple[int], np.dtype[np.float64]]]:
        try:
            timestamp = np.datetime64(row["date"]).astype("datetime64[ns]")
            open_ = float(row["open"])
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            amount = float(row["amount"])
            volume = float(row["volume"]) * float(self._volume_lot_size)
        except Exception as exc:
            raise ValueError(
                f"Daily market snapshot source '{self.name}' parse failure at row {row_index}: {row!r}"
            ) from exc

        if not self._is_valid_snapshot(open_, close, high, low, amount, volume):
            raise ValueError(
                f"Daily market snapshot source '{self.name}' failed sanity checks at row {row_index}: {row!r}"
            )

        vector = np.array([open_, close, high, low, amount, volume], dtype=np.float64)
        return timestamp, vector

    @staticmethod
    def _is_valid_snapshot(open_: float, close: float, high: float, low: float, amount: float, volume: float) -> bool:
        if low < 0.0:
            return False
        if not (low <= open_ <= high):
            return False
        if not (low <= close <= high):
            return False
        if amount < 0.0:
            return False
        if volume < 0.0:
            return False
        return True


__all__ = [
    "DEFAULT_DAILY_MARKET_SNAPSHOT_SCHEMA",
    "DailyMarketSnapshotCSVSource",
    "DailyMarketSnapshotDiagnostics",
    "DailyMarketSnapshotSchema",
]
