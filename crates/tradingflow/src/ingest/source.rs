//! The packaged one-cell event source.

use futures::stream::Stream;

use super::feed::Event;
use crate::Instant;

/// An event source feeding one graph source cell: an initial cell value, a
/// stamped event stream, and a write function applying each event to the cell.
///
/// This is the packaged form of a [`Feed`](super::Feed) that writes into a
/// single source node, consumed by [`Scenario::add_source`](super::Scenario::add_source):
/// the builder allocates a `RefSource<Output>` node holding
/// [`initial`](Self::initial), and the driver applies each merged event to the
/// node's cell via [`write`](Self::write). It is the one packaged entry point
/// on the [`Scenario`](super::Scenario); a feed that writes into several nodes
/// (or none) is a raw [`Feed`](super::Feed) driven through a
/// [`Queue`](super::Queue) directly.
///
/// # Lifecycle
///
/// 1. [`initial`](Self::initial) produces the value the source cell holds
///    before any event (read at graph-build time).
/// 2. [`init`](Self::init) reads the spec by reference and produces the event
///    [`Stream`] plus the initial write [`State`](Self::State). It runs lazily
///    on the driving task at the first [`Session::step`](super::Session::step)
///    (via a [`LazyFeed`](super::LazyFeed)), so it may spawn producer tasks on
///    an async runtime or open connections.
/// 3. [`write`](Self::write) is called per received event payload to update
///    the cell (threading `State`), returning how many logical events the
///    payload represented.
///
/// # Reusability
///
/// `init` takes `&self`, so a single (typically `Clone`) spec can drive
/// multiple sessions, each `init` producing a fresh stream and state. Treat
/// the spec as immutable configuration; per-run state lives in
/// [`State`](Self::State).
pub trait EventSource: Send + 'static {
    /// Stream event type.
    type Event: Send + 'static;
    /// The source cell's value type.
    type Output: Send + Sync + 'static;
    /// Per-run mutable state threaded through every [`write`](Self::write)
    /// call (created by [`init`](Self::init)). Use `()` for stateless sources;
    /// sources that need to remember something across events — e.g. a panel
    /// that clears the previous tick's entries when the timestamp advances —
    /// keep it here.
    type State: Send + 'static;

    /// The value the source cell holds before any event arrives.
    fn initial(&self) -> Self::Output;

    /// Build the event stream and the initial write [`State`](Self::State)
    /// from a borrow of the spec.
    ///
    /// Event stamps must be monotonic non-decreasing; [`Event::frontier`] and
    /// [`Event::now`] work as for any feed (see [`Queue`](super::Queue) for
    /// the merge semantics).
    fn init(
        &self,
    ) -> (
        impl Stream<Item = Event<Self::Event>> + Send + 'static,
        Self::State,
    );

    /// Apply a received payload to the cell, threading `state`, and return
    /// **how many logical events it represents** (for the running session's
    /// event counter).
    ///
    /// A source may batch many events into one stream item (e.g. a panel ships
    /// a whole tick's rows as one `Vec`); `write` then applies them per event
    /// and returns the batch size. A one-event-per-item source returns `1`.
    /// The engine marks the cell's dirty cone regardless of the return.
    fn write(
        state: &mut Self::State,
        event: Self::Event,
        output: &mut Self::Output,
        timestamp: Instant,
    ) -> usize;

    /// Estimated total number of events this source will emit over its
    /// lifetime, or `None` for unbounded/unknown. Advisory — used only for
    /// progress reporting against the session's event counter
    /// ([`Session::total_num_events`](super::Session::total_num_events)). The
    /// default returns `None`.
    fn total_num_events(&self) -> Option<usize> {
        None
    }
}
