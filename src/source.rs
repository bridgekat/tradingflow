use tokio::sync::mpsc;

use super::Output;

/// A data source providing historical and/or real-time data streams.
///
/// Each source maintains two conceptual queues (historical + real-time),
/// implemented as bounded tokio [`mpsc`] channels. `subscribe` consumes
/// the source, starts producing events, and returns the receivers.
///
/// - **Historical channel**: `(timestamp, event)` with pre-recorded
///   timestamps in non-decreasing order. Timestamps will be treated
///   as-is, but must guarantee liveness: if the producer is still
///   running, the channel will eventually receive another event.
/// - **Real-time channel**: `(timestamp, event)` with source-provided
///   timestamps in non-decreasing order. Can wait indefinitely for
///   another event, but the [`Scenario`](crate::Scenario) may adjust
///   timestamps to enforce constraints.
///
/// The output must be an [`Observable`](crate::Observable).
///
/// The [`Scenario`](crate::Scenario) will call [`Source::subscribe`] to start
/// the source, then call [`Source::write`] on each event received from either
/// the historical or the real-time channel.
pub trait Source: Send + 'static {
    /// Implementor-defined channel event type.
    type Event: Send + 'static;

    /// The source's output container.
    ///
    /// Must be an [`Observable`](crate::Observable).
    type Output: Output;

    /// Infer the output element shape.
    fn shape(&self) -> Box<[usize]>;

    /// Provide the initial value for the output observable.
    fn initial(&self) -> Box<[<Self::Output as Output>::Scalar]>;

    /// Start producing events. Creates channels, spawns producers, and
    /// returns the historical and real-time receivers.
    ///
    /// All async work is deferred into the spawned producer tasks;
    /// this method itself is synchronous (but may require a tokio
    /// runtime context for [`tokio::spawn`]).
    fn subscribe(
        self,
    ) -> (
        mpsc::Receiver<(i64, Self::Event)>,
        mpsc::Receiver<(i64, Self::Event)>,
    );

    /// Write an event into the output, or return `false` if no output is
    /// produced.
    fn write(event: Self::Event, output: &mut Self::Output) -> bool;
}
