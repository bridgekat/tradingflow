//! Iterator-based source — feeds events from an arbitrary iterator factory.

use std::rc::Rc;

use tokio::sync::mpsc;

use crate::Instant;
use crate::source::Source;

/// Boxed iterator type produced by an [`IterSource`] factory.
type EventIter<T> = Box<dyn Iterator<Item = (Instant, T)> + Send>;

/// Shared, cheaply-cloned factory that produces fresh event iterators.
///
/// Stored behind `Rc<...>` so the [`IterSource`] spec satisfies `Clone`
/// (refcount bump only) and can drive multiple scenario sessions, each
/// with a freshly built iterator.  `Rc` (rather than `Arc`) is used
/// deliberately: the spec is constructed and consumed on the registering
/// thread, and the produced iterator (which crosses thread boundaries
/// via the tokio task spawned in [`Source::init`]) is itself `Send`.
type IterFactory<T> = Rc<dyn Fn() -> EventIter<T>>;

/// A source driven by a factory of `(timestamp, event)` iterators.
///
/// More flexible than [`ArraySource`](super::ArraySource) — supports lazy
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
            factory: Rc::clone(&self.factory),
            default: self.default.clone(),
            estimated_event_count: self.estimated_event_count,
        }
    }
}

impl<T: Clone + Send + 'static> IterSource<T> {
    /// Create from an iterator factory and a default output value.
    ///
    /// `factory` is called once per [`Source::init`] invocation and must
    /// produce a fresh iterator over `(timestamp, value)` pairs each time.
    /// The factory is shared via `Rc` across clones of the source.
    pub fn new<I, F>(factory: F, default: T) -> Self
    where
        I: Iterator<Item = (Instant, T)> + Send + 'static,
        F: Fn() -> I + 'static,
    {
        Self {
            factory: Rc::new(move || Box::new(factory())),
            default,
            estimated_event_count: None,
        }
    }

    /// Create from a fixed `Vec` of `(timestamp, value)` events.
    ///
    /// The vector is shared across clones via `Rc` and cloned on each
    /// [`Source::init`] call to produce a fresh iterator.  The estimated
    /// event count is set to the vector length automatically.
    pub fn from_vec(events: Vec<(Instant, T)>) -> Self
    where
        T: Default,
    {
        Self::from_vec_with_default(events, T::default())
    }

    /// Like [`from_vec`](Self::from_vec) but with an explicit default
    /// output value.
    pub fn from_vec_with_default(events: Vec<(Instant, T)>, default: T) -> Self {
        let count = events.len();
        let events = Rc::new(events);
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
    /// (e.g. clock sources backed by a `Vec`).  Used only by
    /// [`Scenario::run`](crate::Scenario::run)'s progress callback.
    pub fn with_estimated_count(mut self, count: usize) -> Self {
        self.estimated_event_count = Some(count);
        self
    }
}

impl<T: Clone + Send + 'static> Source for IterSource<T> {
    type Event = T;
    type Output = T;

    fn estimated_event_count(&self) -> Option<usize> {
        self.estimated_event_count
    }

    fn init(
        self,
        _timestamp: Instant,
    ) -> (
        mpsc::Receiver<(Instant, T)>,
        mpsc::Receiver<(Instant, T)>,
        T,
    ) {
        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);

        let iter = (self.factory)();
        tokio::spawn(async move {
            for (ts, value) in iter {
                if hist_tx.send((ts, value)).await.is_err() {
                    break;
                }
            }
        });

        (hist_rx, live_rx, self.default)
    }

    fn write(payload: T, output: &mut T, _timestamp: Instant) -> bool {
        *output = payload;
        true
    }
}

