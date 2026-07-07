use tokio::sync::mpsc;

use crate::Instant;

/// An asynchronous data source that streams timestamped events into a graph
/// source cell.
///
/// # Lifecycle
///
/// 1. [`initial`](Self::initial) produces the value the source cell holds
///    before any event (read at graph-build time).
/// 2. [`init`](Self::init) reads the spec by reference and produces a channel
///    receiver of `(timestamp, event)` (filled by a spawned producer task, in
///    non-decreasing timestamp order) and the initial write
///    [`State`](Self::State).
/// 3. [`write`](Self::write) is called for each received channel item to update
///    the output (threading `State`), returning how many logical events it
///    represented.
///
/// # Reusability
///
/// `init` takes `&self`, so a single spec can drive multiple
/// [`Session`](crate::Session)s — the driver keeps the spec by value and
/// calls `init` against the shared reference on every session start.
/// Implementations should treat the spec as immutable configuration; per-session
/// state lives in [`State`](Self::State) (built fresh by `init`). Clone any
/// field that needs to move into an async producer task (e.g. into
/// [`tokio::spawn`]) explicitly.
///
/// `Send` because the spec moves into its session's lazily-started feed,
/// which lives inside the (Send) event queue until the run begins.
pub trait Source: Send + 'static {
    /// Channel event type.
    type Event: Send + 'static;
    /// Output type.
    type Output: Send + 'static;
    /// Per-source mutable state threaded through every [`write`](Self::write)
    /// call (created by [`init`](Self::init)). Use `()` for stateless sources;
    /// sources that need to remember something across events — e.g. a panel that
    /// clears the previous tick's entries when the timestamp advances — keep it
    /// here.
    type State: Send + 'static;

    /// The value the source cell holds before any event arrives.
    fn initial(&self) -> Self::Output;

    /// Build the event channel receiver and the initial write
    /// [`State`](Self::State) from a borrow of the spec, spawning the
    /// producer task.
    fn init(&self) -> (mpsc::Receiver<(Instant, Self::Event)>, Self::State);

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
