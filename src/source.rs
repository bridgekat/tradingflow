use tokio::sync::mpsc;

use super::store::ElementViewMut;
use super::types::Scalar;

/// A data source providing historical and/or real-time data streams,
/// writing into [`Store`](crate::store::Store) via [`ElementViewMut`].
///
/// Each source maintains two conceptual queues (historical + real-time),
/// implemented as bounded tokio [`mpsc`] channels. `subscribe` consumes
/// the source, starts producing events, and returns the receivers.
///
/// - **Historical channel**: `(timestamp, event)` with pre-recorded
///   timestamps in non-decreasing order.
/// - **Real-time channel**: `(timestamp, event)` with source-provided
///   timestamps in non-decreasing order.
pub trait Source: Send + 'static {
    /// Implementor-defined channel event type.
    type Event: Send + 'static;

    /// The scalar type of the output store.
    type Scalar: Scalar;

    /// Compute the output element shape and default value.
    ///
    /// Returns `(output_shape, default_values)`.
    fn default(&self) -> (Box<[usize]>, Box<[Self::Scalar]>);

    /// Start producing events.  Returns (historical_rx, live_rx).
    fn subscribe(
        self,
    ) -> (
        mpsc::Receiver<(i64, Self::Event)>,
        mpsc::Receiver<(i64, Self::Event)>,
    );

    /// Write an event into the output view, or return `false` if no output
    /// is produced.
    fn write(event: Self::Event, output: ElementViewMut<'_, Self::Scalar>) -> bool;
}
