//! Built-in data sources for the computation graph.
//!
//! Every source in this module implements [`Source`](crate::Source) and is
//! registered into a [`Scenario`](crate::Scenario) via
//! [`Scenario::add_source`](crate::Scenario::add_source). Sources stream events
//! into the graph through an async channel; the event loop
//! ([`Session::run`](crate::Session::run)) merges them in timestamp order.
//!
//! # Data sources
//!
//! - [`ArraySource`] - historical-only source backed by pre-loaded timestamp
//!   and value arrays. Each event carries an `Array<T>`. Requires a tokio
//!   runtime.
//! - [`CsvSource`] - historical-only source that reads a timestamp column and
//!   named value columns from a CSV file on disk. Outputs `Array<f64>`. Requires
//!   a tokio runtime.
//! - [`IterSource`] - source driven by an arbitrary `(timestamp, value)`
//!   iterator. More flexible than `ArraySource`; supports lazy/computed sequences
//!   and arbitrary output types. Requires a tokio runtime.
//! - [`ParquetPanelSource`] - cross-sectional panel over a long-format Parquet
//!   table; emits one wide `[N, K]` cross-section per date. Requires a tokio
//!   runtime.
//! - [`ReportPanelSource`] - panel variant for financial-report long tables,
//!   with point-in-time effective-date alignment. Requires a tokio runtime.
//!
//! # Clock sources
//!
//! Clock sources emit `()` events at specified timestamps and are used as
//! triggers for periodic operators.
//!
//! - [`clock`] - clock from explicit timestamps.  Calendar-aligned schedules
//!   (daily / monthly in a given timezone) are constructed in Python via
//!   `zoneinfo` and passed in as a pre-computed list, keeping the Rust core
//!   free of timezone data.
//!
//! # Sub-modules
//!
//! - [`stocks`] - stock-specific data sources.

pub mod array_source;
pub mod clock;
pub mod csv_source;
pub mod iter_source;
pub mod parquet_panel_source;
pub mod report_panel_source;
pub mod stocks;

pub use array_source::ArraySource;
pub use clock::clock;
pub use csv_source::CsvSource;
pub use iter_source::IterSource;
pub use parquet_panel_source::ParquetPanelSource;
pub use report_panel_source::ReportPanelSource;
