//! The packaged one-cell event source.

use futures::stream::Stream;

use super::feed::Event;
use crate::Instant;
use crate::graph::Value;

/// An event source feeding one graph source cell: an initial cell value, a
/// stamped event stream, and a write function applying each event to the cell.
///
/// This is the packaged form of a [`Feed`](super::Feed) that writes into a
/// single source node, consumed by
/// [`Scenario::add_source`](super::Scenario::add_source): the source names its
/// cell's [`Value`] kind ([`Value`](Self::Value)), the builder allocates a
/// [`ViewSource<Self::Value>`](crate::graph::ViewSource) node holding
/// [`initial`](Self::initial), and the driver applies each merged event to the
/// node's owned cell via the writer [`init`](Self::init) returns. The
/// returned handle is the
/// cell's `ViewPort<Self::Value>` — an `ArrayValue` source wires as an
/// `ArrayPort` view edge, a `Ref<T>` source as a whole-value `RefPort<T>`
/// edge — so it wires into the operator library with no adapter. It is the
/// one packaged entry point on the [`Scenario`](super::Scenario); a feed that
/// writes into several nodes (or none) is a raw [`Feed`](super::Feed) driven
/// through a [`Queue`](super::Queue) directly.
///
/// # Lifecycle
///
/// 1. [`initial`](Self::initial) produces the value the source cell holds
///    before any event (read at graph-build time).
/// 2. [`init`](Self::init) reads the spec by reference and produces the event
///    [`Stream`] plus the **writer** — the closure called per received event
///    payload to update the cell, returning how many logical events the
///    payload represented. `init` runs lazily on the driving task at the first
///    [`Session::step`](super::Session::step) (via a
///    [`LazyFeed`](super::LazyFeed)), so it may spawn producer tasks on an
///    async runtime or open connections.
///
/// # Reusability
///
/// `init` takes `&self`, so a single (typically `Clone`) spec can drive
/// multiple sessions, each `init` producing a fresh stream and writer. Treat
/// the spec as immutable configuration; per-run state lives in the writer's
/// (and the stream's) captures — e.g. a panel source captures the rows the
/// previous tick dirtied, so it can clear exactly those when the timestamp
/// advances.
pub trait EventSource: Send + 'static
where
    for<'a> <Self::Value as Value>::View<'a>: Copy + Send + Sync,
{
    /// Stream event type.
    type Event: Send + 'static;
    /// The [`Value`] kind of the cell this source feeds: the cell holds a
    /// [`Value::Owned`] and the handle speaks `ViewPort<Self::Value>`. An
    /// array source names [`ArrayValue`](crate::operators::ArrayValue); a
    /// whole-value payload (a pulse's `()`, an event batch `Vec<E>`) names
    /// [`Ref<T>`](crate::graph::Ref).
    type Value: Value;

    /// The value the source cell holds before any event arrives.
    fn initial(&self) -> <Self::Value as Value>::Owned;

    /// Build the event stream and the writer from a borrow of the spec.
    ///
    /// The writer applies one received payload to the cell and returns **how
    /// many logical events it represents** (for the running session's event
    /// counter). A source may batch many events into one stream item (e.g. a
    /// panel ships a whole tick's rows as one `Vec`); the writer then applies
    /// them per event and returns the batch size. A one-event-per-item source
    /// returns `1`. The engine marks the cell's dirty cone regardless of the
    /// return. Per-run mutable state lives in the writer's captures.
    ///
    /// Event stamps must be monotonic non-decreasing; [`Event::frontier`] and
    /// [`Event::now`] work as for any feed (see [`Queue`](super::Queue) for
    /// the merge semantics).
    #[expect(clippy::type_complexity)]
    fn init(
        &self,
    ) -> (
        impl Stream<Item = Event<Self::Event>> + Send + 'static,
        impl FnMut(Self::Event, &mut <Self::Value as Value>::Owned, Instant) -> usize + Send + 'static,
    );

    /// Estimated total number of events this source will emit over its
    /// lifetime, or `None` for unbounded/unknown. Advisory — used only for
    /// progress reporting against the session's event counter
    /// ([`Session::total_num_events`](super::Session::total_num_events)). The
    /// default returns `None`.
    fn total_num_events(&self) -> Option<usize> {
        None
    }
}
