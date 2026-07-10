//! Iterator-based source - feeds events from an arbitrary iterator factory.

use std::sync::Arc;

use futures::stream::Stream;
use tokio::sync::mpsc;

use flowgraph::ingest::{Event, EventSource};

use super::receiver_stream;
use crate::Instant;

/// Boxed iterator type produced by an [`IterSource`] factory.
type EventIter<T> = Box<dyn Iterator<Item = (Instant, T)> + Send>;

/// Shared, cheaply-cloned factory that produces fresh event iterators.
///
/// Stored behind `Arc<...>` so the [`IterSource`] spec satisfies `Clone`
/// (refcount bump only) and can drive multiple scenario sessions, each with a
/// freshly built iterator — and `Send`, as [`EventSource`] requires (the spec
/// moves into the session's lazily-started feed).
type IterFactory<T> = Arc<dyn Fn() -> EventIter<T> + Send + Sync>;

/// A source driven by a factory of `(timestamp, event)` iterators.
///
/// More flexible than [`ArraySource`](super::ArraySource) - supports lazy
/// or computed timestamp sequences, and arbitrary output types.  The
/// factory is invoked once per scenario session to produce a fresh
/// iterator; the `IterSource` spec itself is `Clone` and reusable.
///
/// The iterator is drained by a spawned tokio task with bounded
/// back-pressure.
///
/// Requires a tokio runtime to be active when added to a scenario.
pub struct IterSource<T: Clone + Send + 'static> {
    factory: IterFactory<T>,
    default: T,
    estimated_event_count: Option<usize>,
}

impl<T: Clone + Send + 'static> Clone for IterSource<T> {
    fn clone(&self) -> Self {
        Self {
            factory: Arc::clone(&self.factory),
            default: self.default.clone(),
            estimated_event_count: self.estimated_event_count,
        }
    }
}

impl<T: Clone + Send + 'static> IterSource<T> {
    /// Create from an iterator factory and a default output value.
    ///
    /// `factory` is called once per [`EventSource::init`] invocation and must
    /// produce a fresh iterator over `(timestamp, value)` pairs each time.
    /// The factory is shared via `Arc` across clones of the source.
    pub fn new<I, F>(factory: F, default: T) -> Self
    where
        I: Iterator<Item = (Instant, T)> + Send + 'static,
        F: Fn() -> I + Send + Sync + 'static,
    {
        Self {
            factory: Arc::new(move || Box::new(factory())),
            default,
            estimated_event_count: None,
        }
    }

    /// Create from a fixed `Vec` of `(timestamp, value)` events.
    ///
    /// The vector is shared across clones via `Arc` and cloned on each
    /// [`EventSource::init`] call to produce a fresh iterator.  The estimated
    /// event count is set to the vector length automatically.
    pub fn from_vec(events: Vec<(Instant, T)>) -> Self
    where
        T: Default + Sync,
    {
        Self::from_vec_with_default(events, T::default())
    }

    /// Like [`from_vec`](Self::from_vec) but with an explicit default
    /// output value.
    pub fn from_vec_with_default(events: Vec<(Instant, T)>, default: T) -> Self
    where
        T: Sync,
    {
        let count = events.len();
        let events = Arc::new(events);
        Self::new(
            move || {
                let snapshot: Vec<(Instant, T)> = (*events).clone();
                snapshot.into_iter()
            },
            default,
        )
        .with_estimated_count(count)
    }

    /// Advertise an estimated total event count.
    ///
    /// Call this when the iterator length is known at construction time
    /// (e.g. clock sources backed by a `Vec`).  Used only for progress
    /// reporting ([`Graph::estimated_event_count`](flowgraph::ingest::Graph::estimated_event_count)).
    pub fn with_estimated_count(mut self, count: usize) -> Self {
        self.estimated_event_count = Some(count);
        self
    }
}

impl<T: Clone + Send + Sync + 'static> EventSource<Instant> for IterSource<T> {
    type Event = T;
    type Output = T;
    type State = ();

    fn estimated_event_count(&self) -> Option<usize> {
        self.estimated_event_count
    }

    fn initial(&self) -> T {
        self.default.clone()
    }

    fn init(&self) -> (impl Stream<Item = Event<Instant, T>> + Send + 'static, ()) {
        let (hist_tx, hist_rx) = mpsc::channel(64);

        let iter = (self.factory)();
        tokio::spawn(async move {
            for (ts, value) in iter {
                if hist_tx.send((ts, value)).await.is_err() {
                    break;
                }
            }
        });

        (receiver_stream(hist_rx), ())
    }

    fn write(_state: &mut (), payload: T, output: &mut T, _timestamp: Instant) -> usize {
        *output = payload;
        1
    }
}
