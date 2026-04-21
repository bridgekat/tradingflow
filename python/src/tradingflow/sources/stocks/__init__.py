"""Stock-specific data sources.

Sources tailored to quirks of equity data that don't fit cleanly into
the generic sources in the parent
[`tradingflow.sources`][tradingflow.sources] module.  All sources in
this module are [`NativeSource`][tradingflow.source.NativeSource]
subclasses dispatched entirely to Rust.

- [`FinancialReportSource`][tradingflow.sources.stocks.financial_report_source.FinancialReportSource] —
  historical source for financial-report CSVs.  Unlike price data,
  financial reports have two relevant timestamps per row: the *report
  date* (the period the numbers describe — e.g. 2024-Q1) and the
  *notice date* (when the filing was actually released, which is when
  the market first knew about it).  This source emits events at the
  notice date but carries both dates in the payload, so downstream
  operators can use the report date for alignment without looking
  ahead.
"""

from .financial_report_source import FinancialReportSource

__all__ = [
    "FinancialReportSource",
]
