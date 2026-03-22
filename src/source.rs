use tokio::sync::mpsc;

/// A data source providing historical and/or real-time event streams.
///
/// Each source maintains two conceptual queues (historical + real-time),
/// implemented as bounded tokio [`mpsc`] channels.  [`init`](Source::init)
/// consumes the source, starts producing events, and returns the receivers
/// together with the initial output value.
///
/// - **Historical channel**: `(timestamp, event)` with pre-recorded
///   timestamps in non-decreasing order.
/// - **Real-time channel**: `(timestamp, event)` with source-provided
///   timestamps in non-decreasing order.
///
/// Sources that use [`tokio::spawn`] in `init()` require a tokio runtime
/// to be active when [`Scenario::add_source`] is called.
pub trait Source: Send + 'static {
    /// Implementor-defined channel event type.
    type Event: Send + 'static;

    /// The output value type held by this source's node.
    type Output: Send + 'static;

    /// Consume the source, start producing events, and return
    /// `(historical_rx, live_rx, initial_output)`.
    ///
    /// `timestamp` is `i64::MIN` (reserved for future use).
    fn init(
        self,
        timestamp: i64,
    ) -> (
        mpsc::Receiver<(i64, Self::Event)>,
        mpsc::Receiver<(i64, Self::Event)>,
        Self::Output,
    );

    /// Write an event into the output, or return `false` if no output
    /// is produced.
    fn write(event: Self::Event, output: &mut Self::Output, timestamp: i64) -> bool;
}
