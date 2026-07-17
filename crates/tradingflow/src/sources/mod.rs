//! Built-in data sources for the computation graph.
//!
//! Every source in this module implements [`EventSource`](crate::ingest::EventSource)
//! and is registered into a [`Scenario`](crate::Scenario) via
//! [`Scenario::source`](crate::Scenario::source). Sources stream
//! events into the graph through an async channel bridged by
//! [`receiver_stream`]; the event loop ([`Session::run`](crate::Session::run))
//! merges them in timestamp order.
//!
//! **Every source has a lowercase free constructor** — [`array_source`],
//! [`iter_source`], [`parquet_panel_source`], … — taking the source's
//! parameters and leaving `T` / `N` to be inferred from the wiring at
//! [`Scenario::source`](crate::Scenario::source), exactly as the
//! [operators](crate::operators) do. Prefer them to the inherent
//! `Source::<T, N>::new(..)` forms. The `with_*` builders stay methods, chained
//! onto the constructor.
//!
//! # [`basic`] — in-memory sources
//!
//! - [`array_source`] ([`ArraySource`]) - historical-only source backed by
//!   pre-loaded timestamp and value arrays. Each event carries an `Array<T>`.
//! - [`iter_source`] / [`vec_source`] ([`IterSource`]) - source driven by an
//!   arbitrary `(timestamp, value)` iterator. More flexible than `ArraySource`;
//!   supports lazy/computed sequences and arbitrary output types.
//! - [`pulse`] - `()` triggers from explicit timestamps, the clock the gated
//!   operators fire on. Calendar-aligned schedules (daily / monthly in a given
//!   timezone) are constructed in Python via `zoneinfo` and passed in as a
//!   pre-computed list, keeping the Rust core free of timezone data.
//!
//! # [`panel`] — cross-sectional Parquet panels
//!
//! - [`parquet_panel_source`] ([`ParquetPanelSource`]) - cross-sectional panel
//!   over a long-format Parquet table; emits one wide `[N, K]` cross-section per
//!   date.
//! - [`parquet_financial_report_panel_source`]
//!   ([`ParquetFinancialReportPanelSource`]) - panel variant for
//!   financial-report long tables, with point-in-time effective-date alignment.
//!
//! All of them stream through a spawned tokio task, so a tokio runtime must be
//! active when they are added to a scenario.

use futures::stream::Stream;
use tokio::sync::mpsc;

use crate::data::Instant;
use crate::ingest::Event;

pub mod basic;
pub mod panel;

pub use basic::{ArraySource, IterSource, array_source, iter_source, pulse, vec_source};
pub use panel::{
    ParquetFinancialReportPanelSource, ParquetPanelSource, parquet_financial_report_panel_source,
    parquet_panel_source,
};

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
