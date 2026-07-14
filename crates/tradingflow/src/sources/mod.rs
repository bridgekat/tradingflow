//! Built-in data sources for the computation graph.
//!
//! Every source in this module implements [`EventSource`](crate::ingest::EventSource)
//! and is registered into a [`Scenario`](crate::Scenario) via
//! [`Scenario::add_source`](crate::Scenario::add_source). Sources stream
//! events into the graph through an async channel bridged by
//! [`receiver_stream`]; the event loop ([`Session::run`](crate::Session::run))
//! merges them in timestamp order.
//!
//! # Data sources
//!
//! - [`ArraySource`] - historical-only source backed by pre-loaded timestamp
//!   and value arrays. Each event carries an `Array<T>`. Requires a tokio
//!   runtime.
//! - [`IterSource`] - source driven by an arbitrary `(timestamp, value)`
//!   iterator. More flexible than `ArraySource`; supports lazy/computed sequences
//!   and arbitrary output types. Requires a tokio runtime.
//! - [`ParquetPanelSource`] - cross-sectional panel over a long-format Parquet
//!   table; emits one wide `[N, K]` cross-section per date. Requires a tokio
//!   runtime.
//! - [`ParquetFinancialReportPanelSource`] - panel variant for financial-report long tables,
//!   with point-in-time effective-date alignment. Requires a tokio runtime.
//!
//! # Pulse sources
//!
//! Pulse sources emit `()` events at specified timestamps and are used as
//! triggers for periodic (clock-gated) operators.
//!
//! - [`pulse()`] - `()` triggers from explicit timestamps.  Calendar-aligned schedules
//!   (daily / monthly in a given timezone) are constructed in Python via
//!   `zoneinfo` and passed in as a pre-computed list, keeping the Rust core
//!   free of timezone data.

use futures::stream::Stream;
use tokio::sync::mpsc;

use crate::Instant;
use crate::ingest::Event;

pub mod array_source;
pub mod iter_source;
pub mod parquet_financial_report_panel_source;
pub mod parquet_panel_source;
pub mod pulse;

pub use array_source::ArraySource;
pub use iter_source::IterSource;
pub use parquet_financial_report_panel_source::ParquetFinancialReportPanelSource;
pub use parquet_panel_source::ParquetPanelSource;
pub use pulse::pulse;

/// Adapt a producer channel — `(timestamp, event)` items in non-decreasing
/// timestamp order, closed when the producer finishes — into the
/// explicitly-stamped event stream
/// [`EventSource::init`](crate::ingest::EventSource::init) returns.
///
/// The bridge every source in this module uses: `init` spawns a tokio producer
/// task feeding the channel's sender (bounded, so the producer back-pressures
/// against the event loop) and returns `receiver_stream(rx)`. Custom sources
/// built on tokio tasks can reuse it.
pub fn receiver_stream<E: Send + 'static>(
    rx: mpsc::Receiver<(Instant, E)>,
) -> impl Stream<Item = Event<E>> + Send + 'static {
    futures::stream::unfold(rx, |mut rx| async move {
        let (ts, event) = rx.recv().await?;
        Some((Event::at(ts, event), rx))
    })
}
