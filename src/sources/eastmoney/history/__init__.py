"""EastMoney historical data source adapters.

This package contains source adapters built for EastMoney raw history data
layouts and normalization rules.

Public API
----------
DailyMarketSnapshotCSVSource
    Raw daily price history adapter.
FinancialReportCSVSource
    Raw balance/income statement history adapter.
"""

from .daily_market_snapshot import (
    DEFAULT_DAILY_MARKET_SNAPSHOT_SCHEMA,
    DailyMarketSnapshotCSVSource,
    DailyMarketSnapshotDiagnostics,
    DailyMarketSnapshotSchema,
)
from .financial_reports import (
    BALANCE_SHEET_MAPPING_PROFILE,
    BALANCE_SHEET_SCHEMA,
    INCOME_STATEMENT_MAPPING_PROFILE,
    INCOME_STATEMENT_SCHEMA,
    FinancialReportCSVSource,
    FinancialReportDiagnostics,
    FinancialReportKind,
    FinancialReportMappingProfile,
    FinancialReportRow,
    FinancialReportSchema,
    default_mapping_profile,
    default_schema,
    normalize_financial_report_rows,
)

__all__ = [
    "BALANCE_SHEET_MAPPING_PROFILE",
    "BALANCE_SHEET_SCHEMA",
    "DEFAULT_DAILY_MARKET_SNAPSHOT_SCHEMA",
    "DailyMarketSnapshotCSVSource",
    "DailyMarketSnapshotDiagnostics",
    "DailyMarketSnapshotSchema",
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
