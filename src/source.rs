use tokio::sync::mpsc;

use crate::Instant;

/// An asynchronous data source that streams timestamped events into a graph
/// source cell.
///
/// # Lifecycle
///
/// 1. [`init`](Self::init) reads the spec by reference and produces a channel
///    receiver of `(timestamp, event)` (filled by a spawned producer task, in
///    non-decreasing timestamp order), the initial [`Output`](Self::Output), and
///    the initial write [`State`](Self::State).
/// 2. [`write`](Self::write) is called for each received channel item to update
///    the output (threading `State`), returning how many logical events it
///    represented.
///
/// # Reusability
///
/// `init` takes `&self`, so a single spec can drive multiple
/// [`Session`](crate::Session)s — the driver keeps the spec by value and
/// calls `init` against the shared reference on every session start.
/// Implementations should treat the spec as immutable configuration; per-session
/// state lives in [`Output`](Self::Output) / [`State`](Self::State) (built fresh
/// by `init`). Clone any field that needs to move into an async driver task
/// (e.g. into [`tokio::spawn`]) explicitly.
pub trait Source: 'static {
    /// Channel event type.
    type Event: Send + 'static;
    /// Output type.
    type Output: Send + 'static;
    /// Per-source mutable state threaded through every [`write`](Self::write)
    /// call (created by [`init`](Self::init)). Use `()` for stateless sources;
    /// sources that need to remember something across events — e.g. a panel that
    /// clears the previous tick's entries when the timestamp advances — keep it
    /// here. Lives on the driver thread, so no `Sync` bound.
    type State: Send + 'static;

    /// Build the event channel receiver, the initial output, and the initial
    /// write [`State`](Self::State) from a borrow of the spec.
    fn init(
        &self,
        timestamp: Instant,
    ) -> (mpsc::Receiver<(Instant, Self::Event)>, Self::Output, Self::State);

    /// Apply a received channel item to the output, threading `state`, and return
    /// **how many logical events it represents** (for the run's event count).
    ///
    /// A source may batch many events into one channel item (e.g. a panel ships a
    /// whole tick's rows as one `Vec`); `write` then applies them **per event**
    /// (the per-event state logic still runs for each) and returns the batch size.
    /// A one-event-per-item source returns `1`. The engine marks the output cell's
    /// cone regardless of the return.
    fn write(
        state: &mut Self::State,
        event: Self::Event,
        output: &mut Self::Output,
        timestamp: Instant,
    ) -> usize;

    /// Estimated total number of events this source will emit over its
    /// lifetime. `None` for unbounded sources.
    ///
    /// Used only for progress reporting via
    /// [`Session::run`](crate::Session::run)'s `on_flush` callback —
    /// treated as advisory. The default returns `None`.
    fn estimated_event_count(&self) -> Option<usize> {
        None
    }
}
