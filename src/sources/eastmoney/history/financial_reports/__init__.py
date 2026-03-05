"""EastMoney history adapters and utilities for raw financial report CSVs."""

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
