"""EastMoney history adapters and utilities for raw financial report CSVs.

This package normalizes raw EastMoney balance-sheet and income-statement CSV
files into fixed-order float64 vectors ready for source emission.

Public API
----------
FinancialReportKind
    Literal type alias: ``"balance_sheet"`` or ``"income_statement"``.
FinancialReportSchema
    Canonical field ordering for one report kind; constructed via
    :meth:`FinancialReportSchema.from_field_ids`.
BALANCE_SHEET_SCHEMA, INCOME_STATEMENT_SCHEMA
    Default canonical schemas for each report kind.
default_schema
    Returns the default :class:`FinancialReportSchema` for a given kind.
FinancialReportMappingProfile
    Normalization rules (column aliases, net items, sign conventions) for one
    report kind.
BALANCE_SHEET_MAPPING_PROFILE, INCOME_STATEMENT_MAPPING_PROFILE
    Default mapping profiles for each report kind.
default_mapping_profile
    Returns the default :class:`FinancialReportMappingProfile` for a given kind.
FinancialReportRow
    One normalized report row (dates + float64 vector + error flag) ready for
    source emission.
FinancialReportDiagnostics
    Normalization diagnostics: unknown columns, equation failures, row counts.
normalize_financial_report_rows
    Converts a list of raw CSV row dicts into normalized rows and diagnostics.
FinancialReportCSVSource
    Payload-timestamp :class:`~src.source.Source` adapter for raw financial
    report CSV files.
"""

from __future__ import annotations

from .normalizer import FinancialReportDiagnostics, FinancialReportRow, normalize_financial_report_rows
from .rules import (
    BALANCE_SHEET_MAPPING_PROFILE,
    FinancialReportMappingProfile,
    default_mapping_profile,
    INCOME_STATEMENT_MAPPING_PROFILE,
)
from .schema import (
    BALANCE_SHEET_SCHEMA,
    FinancialReportKind,
    FinancialReportSchema,
    default_schema,
    INCOME_STATEMENT_SCHEMA,
)
from .source import FinancialReportCSVSource

__all__ = [
    "BALANCE_SHEET_MAPPING_PROFILE",
    "BALANCE_SHEET_SCHEMA",
    "FinancialReportCSVSource",
    "FinancialReportDiagnostics",
    "FinancialReportKind",
    "FinancialReportMappingProfile",
    "FinancialReportRow",
    "FinancialReportSchema",
    "INCOME_STATEMENT_MAPPING_PROFILE",
    "INCOME_STATEMENT_SCHEMA",
    "default_mapping_profile",
    "default_schema",
    "normalize_financial_report_rows",
]
