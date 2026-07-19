//! Built-in data sources.

use futures::stream::Stream;
use tokio::sync::mpsc;

use crate::data::Instant;
use crate::graph::Event;

pub mod basic;
pub mod panel;

fn receiver_stream<E: Send + 'static>(
    rx: mpsc::Receiver<(E, Instant)>,
) -> impl Stream<Item = Event<E, Instant>> + Send + 'static {
    futures::stream::unfold(rx, |mut rx| async move {
        let (ts, event) = rx.recv().await?;
        Some((Event::at(ts, event), rx))
    })
}
