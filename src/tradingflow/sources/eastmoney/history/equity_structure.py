"""EastMoney history adapter for raw equity structure CSV files.

Provides :class:`EquityStructureCSVSource`, a historical-only source that
reads raw equity structure CSVs and emits ``TOTAL_SHARES`` (float64 scalar)
at each ``END_DATE``.
"""

from __future__ import annotations

import csv
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ....source import Source, empty_live_gen


@dataclass(slots=True, frozen=True)
class EquityStructureDiagnostics:
    """Diagnostics for equity structure parsing."""

    dropped_rows: int
    total_rows: int
    emitted_rows: int

    @staticmethod
    def empty() -> EquityStructureDiagnostics:
        """Returns an empty diagnostics record."""
        return EquityStructureDiagnostics(dropped_rows=0, total_rows=0, emitted_rows=0)


class EquityStructureCSVSource(Source[tuple[()], np.float64]):
    """Historical source for raw equity structure CSV files.

    Expected raw columns: ``END_DATE``, ``TOTAL_SHARES``.

    Output is a scalar float64 representing the total number of shares
    at each equity change date.
    """

    __slots__ = ("_path", "_strict_row_checks", "_diagnostics")

    _path: Path
    _strict_row_checks: bool
    _diagnostics: EquityStructureDiagnostics

    def __init__(
        self,
        path: str | Path,
        *,
        strict_row_checks: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__((), np.dtype(np.float64), name=name)
        self._path = Path(path)
        self._strict_row_checks = strict_row_checks
        self._diagnostics = EquityStructureDiagnostics.empty()

    @property
    def diagnostics(self) -> EquityStructureDiagnostics:
        """Latest parsing diagnostics."""
        return self._diagnostics

    def subscribe(self) -> tuple[AsyncIterator[tuple[np.datetime64, Any]], AsyncIterator[Any]]:
        """Returns a ``(historical, live)`` iterator pair; the live iterator is empty."""
        return self._historical_gen(), empty_live_gen()

    async def _historical_gen(self) -> AsyncIterator[tuple[np.datetime64, Any]]:
        required_columns = {"END_DATE", "TOTAL_SHARES"}

        dropped_rows = 0
        total_rows = 0
        emitted_rows = 0

        entries: list[tuple[np.datetime64, float]] = []

        with self._path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or ())
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise ValueError(
                    f"Equity structure source '{self.name}' is missing required columns: {missing}"
                )

            for row_index, row in enumerate(reader, start=2):
                total_rows += 1
                try:
                    end_date_raw = row["END_DATE"].strip()
                    total_shares_raw = row["TOTAL_SHARES"].strip()
                    if not end_date_raw or not total_shares_raw:
                        raise ValueError("empty field")
                    timestamp = np.datetime64(end_date_raw.split(" ")[0]).astype("datetime64[ns]")
                    total_shares = float(total_shares_raw)
                    if total_shares <= 0:
                        raise ValueError(f"non-positive total shares: {total_shares}")
                except (ValueError, KeyError) as exc:
                    dropped_rows += 1
                    if self._strict_row_checks:
                        raise ValueError(
                            f"Equity structure source '{self.name}' parse failure at row {row_index}: {row!r}"
                        ) from exc
                    continue
                entries.append((timestamp, total_shares))

        # Sort by timestamp and deduplicate (keep last entry per date)
        entries.sort(key=lambda e: e[0])
        deduplicated: list[tuple[np.datetime64, float]] = []
        for ts, val in entries:
            if deduplicated and deduplicated[-1][0] == ts:
                deduplicated[-1] = (ts, val)
            else:
                deduplicated.append((ts, val))

        for ts, val in deduplicated:
            emitted_rows += 1
            yield ts, np.float64(val)

        self._diagnostics = EquityStructureDiagnostics(
            dropped_rows=dropped_rows,
            total_rows=total_rows,
            emitted_rows=emitted_rows,
        )


__all__ = [
    "EquityStructureCSVSource",
    "EquityStructureDiagnostics",
]
