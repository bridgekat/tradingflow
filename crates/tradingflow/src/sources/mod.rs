//! Built-in data sources.

use futures::stream::Stream;
use tokio::sync::mpsc;

use crate::data::Instant;
use crate::ingest::Event;

pub mod basic;
pub mod panel;

/// Adapt a producer channel — `(timestamp, event)` items in non-decreasing
/// timestamp order, closed when the producer finishes — into the
/// explicitly-stamped event stream
/// [`EventSource::init`](crate::ingest::EventSource::init) returns.
///
/// The bridge every source in this module uses: `init` spawns a tokio producer
/// task feeding the channel's sender (bounded, so the producer back-pressures
/// against the event loop) and returns `receiver_stream(rx)`. Custom sources
/// built on tokio tasks can reuse it.
fn receiver_stream<E: Send + 'static>(
    rx: mpsc::Receiver<(Instant, E)>,
) -> impl Stream<Item = Event<E>> + Send + 'static {
    futures::stream::unfold(rx, |mut rx| async move {
        let (ts, event) = rx.recv().await?;
        Some((Event::at(ts, event), rx))
    })
}
