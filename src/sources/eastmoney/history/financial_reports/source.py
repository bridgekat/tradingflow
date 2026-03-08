"""CSV source adapter for normalized EastMoney financial report vectors."""

from __future__ import annotations

import csv
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import numpy as np

from .....source import Source, empty_live_gen
from .normalizer import FinancialReportDiagnostics, FinancialReportRow, normalize_financial_report_rows
from .rules import FinancialReportMappingProfile, default_mapping_profile
from .schema import FinancialReportKind, FinancialReportSchema, default_schema


class FinancialReportCSVSource(Source[tuple[int], np.float64]):
    """Historical source for raw financial report CSV files.

    The source reads one raw report CSV, normalizes rows into a canonical
    fixed-order vector using the provided schema and mapping profile, then
    emits updates at ``relevance_date = max(report_date, notice_date)``.
    """

    __slots__ = (
        "_path",
        "_kind",
        "_schema",
        "_mapping_profile",
        "_symbol",
        "_strict_unknown_columns",
        "_strict_equation_check",
        "_diagnostics",
        "_normalized_rows",
    )

    _path: Path
    _kind: FinancialReportKind
    _schema: FinancialReportSchema
    _mapping_profile: FinancialReportMappingProfile
    _symbol: str | None
    _strict_unknown_columns: bool
    _strict_equation_check: bool
    _diagnostics: FinancialReportDiagnostics
    _normalized_rows: tuple[FinancialReportRow, ...] | None

    def __init__(
        self,
        path: str | Path,
        *,
        kind: FinancialReportKind,
        schema: FinancialReportSchema | None = None,
        mapping_profile: FinancialReportMappingProfile | None = None,
        symbol: str | None = None,
        strict_unknown_columns: bool = False,
        strict_equation_check: bool = False,
        name: str | None = None,
    ) -> None:
        schema_resolved = schema or default_schema(kind)
        shape = cast(tuple[int], (len(schema_resolved.field_ids),))
        super().__init__(shape, np.dtype(np.float64), name=name)
        self._path = Path(path)
        self._kind = kind
        self._schema = schema_resolved
        self._mapping_profile = mapping_profile or default_mapping_profile(kind)
        self._symbol = symbol
        self._strict_unknown_columns = strict_unknown_columns
        self._strict_equation_check = strict_equation_check
        self._diagnostics = FinancialReportDiagnostics.empty()
        self._normalized_rows = None

    @property
    def schema(self) -> FinancialReportSchema:
        """Canonical schema used by this source."""
        return self._schema

    @property
    def diagnostics(self) -> FinancialReportDiagnostics:
        """Latest normalization diagnostics."""
        return self._diagnostics

    def subscribe(self) -> tuple[AsyncIterator[tuple[np.datetime64, Any]], AsyncIterator[Any]]:
        """Returns a ``(historical, live)`` iterator pair; the live iterator is empty."""
        return self._historical_gen(), empty_live_gen()

    async def _historical_gen(self) -> AsyncIterator[tuple[np.datetime64, Any]]:
        if self._normalized_rows is None:
            rows = self._read_rows()
            normalized_rows, diagnostics = normalize_financial_report_rows(
                rows,
                kind=self._kind,
                schema=self._schema,
                mapping_profile=self._mapping_profile,
                symbol=self._symbol,
                strict_unknown_columns=self._strict_unknown_columns,
                strict_equation_check=self._strict_equation_check,
            )
            self._normalized_rows = tuple(normalized_rows)
            self._diagnostics = diagnostics

        for row in self._normalized_rows:
            yield row.relevance_date, row.values

    def _read_rows(self) -> list[dict[str, str]]:
        with self._path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return [dict(row) for row in reader]


__all__ = [
    "FinancialReportCSVSource",
]
