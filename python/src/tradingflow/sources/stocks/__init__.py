"""Stock-specific data sources.

All sources in this module are [`NativeSource`][tradingflow.source.NativeSource]
subclasses dispatched entirely to Rust.

- [`FinancialReportSource`][tradingflow.sources.stocks.FinancialReportSource] --
  historical source for financial report CSVs with separate report-date and
  notice-date columns
"""

from .financial_report_source import FinancialReportSource

__all__ = [
    "FinancialReportSource",
]
