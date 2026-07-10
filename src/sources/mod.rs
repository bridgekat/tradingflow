//! Built-in data sources for the computation graph.
//!
//! Every source in this module implements [`flowgraph::ingest::EventSource`]
//! and is registered into a [`Scenario`](crate::Scenario) via
//! [`Builder::add_source`](flowgraph::ingest::Builder::add_source). Sources
//! stream events into the graph through an async channel bridged by
//! [`receiver_stream`]; the event loop ([`Graph::run`](flowgraph::ingest::Graph::run))
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
//! # Clock sources
//!
//! Clock sources emit `()` events at specified timestamps and are used as
//! triggers for periodic operators.
//!
//! - [`clock`] - clock from explicit timestamps.  Calendar-aligned schedules
//!   (daily / monthly in a given timezone) are constructed in Python via
//!   `zoneinfo` and passed in as a pre-computed list, keeping the Rust core
//!   free of timezone data.

use futures::stream::Stream;
use tokio::sync::mpsc;

use flowgraph::ingest::Event;

use crate::Instant;

pub mod array_source;
pub mod clock;
pub mod iter_source;
pub mod parquet_panel_source;
pub mod parquet_financial_report_panel_source;

pub use array_source::ArraySource;
pub use clock::clock;
pub use iter_source::IterSource;
pub use parquet_panel_source::ParquetPanelSource;
pub use parquet_financial_report_panel_source::ParquetFinancialReportPanelSource;

/// Adapt a producer channel — `(timestamp, event)` items in non-decreasing
/// timestamp order, closed when the producer finishes — into the
/// explicitly-stamped event stream
/// [`EventSource::init`](flowgraph::ingest::EventSource::init) returns.
///
/// The bridge every source in this module uses: `init` spawns a tokio producer
/// task feeding the channel's sender (bounded, so the producer back-pressures
/// against the event loop) and returns `receiver_stream(rx)`. Custom sources
/// built on tokio tasks can reuse it.
pub fn receiver_stream<E: Send + 'static>(
    rx: mpsc::Receiver<(Instant, E)>,
) -> impl Stream<Item = Event<Instant, E>> + Send + 'static {
    futures::stream::unfold(rx, |mut rx| async move {
        let (ts, event) = rx.recv().await?;
        Some((Event::at(ts, event), rx))
    })
}
