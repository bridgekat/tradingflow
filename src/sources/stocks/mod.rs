//! Stock-specific data sources.
//!
//! Sources in this module handle domain-specific data formats for equity
//! market data.
//!
//! - [`FinancialReportSource`] — historical-only source for financial report
//!   CSVs with separate report-date and notice-date columns. Computes event
//!   timestamps as `max(report_date, notice_date)` and optionally prepends
//!   `[year, day_of_year]` metadata for downstream annualisation.

pub mod financial_report_source;

pub use financial_report_source::FinancialReportSource;
