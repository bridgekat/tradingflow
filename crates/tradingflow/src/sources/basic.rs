//! The basic in-memory sources: pre-loaded arrays, arbitrary iterators, and
//! bare clock pulses.
//!
//! * [`ArraySource`] — historical-only replay of a [`Series`], each event an
//!   `Array<T, N>`.
//! * [`IterSource`] — driven by a factory of `(timestamp, value)` iterators;
//!   more flexible than [`ArraySource`], supporting lazy / computed sequences
//!   and arbitrary event types.
//! * [`pulse()`] — a [`PulseSource`] of bare `Val<()>` triggers, the clock the
//!   gated operators fire on.
//!
//! All three drain into the graph through a spawned tokio task with bounded
//! back-pressure, so a tokio runtime must be active when they are added to a
//! scenario.
//!
//! # Pulses
//!
//! The output node holds `()` (zero-sized, purely a trigger signal); pulse
//! handles carry no data. Clock-gated operators ([`Clocked`],
//! [`ResampleClocked`], and the since-inception metrics) take such a pulse as
//! their leading input port and fire only on its notify bit.
//!
//! Not to be confused with [`WallClock`](crate::WallClock), which drives the
//! event loop, or with the ambient event time (the graph-level context every
//! operator's `compute` is handed).
//!
//! Calendar-aligned schedules (daily / monthly in a given timezone) are
//! generated on the Python side via `zoneinfo` and passed to [`pulse`] as a
//! pre-computed list.  Keeping calendar/timezone logic in Python lets the
//! Rust core stay free of `chrono` / `chrono-tz`.
//!
//! [`Clocked`]: crate::operators::structural::Clocked
//! [`ResampleClocked`]: crate::operators::structural::ResampleClocked

use std::sync::Arc;

use futures::stream::Stream;
use tokio::sync::mpsc;

use super::receiver_stream;
use crate::data::{Array, Instant, Scalar, Series};
use crate::graph::{Event, Ref, Source, Val};
use crate::ports::ArrayPass;

// ===========================================================================
// ArraySource — historical replay of a pre-loaded Series.
// ===========================================================================

/// Historical-only source backed by pre-loaded timestamp and value arrays.
///
/// Each event carries an `Array<T>` value.  The event channel is filled by a
/// spawned tokio task with bounded back-pressure.
///
/// Requires a tokio runtime to be active when added to a scenario.
#[derive(Clone)]
pub struct ArraySource<T: Scalar, const N: usize> {
    series: Series<T, N>,
    default: Array<T, N>,
}

impl<T: Scalar, const N: usize> ArraySource<T, N> {
    /// Create from timestamp and flat value arrays.
    ///
    /// `values.len()` must equal `timestamps.len() * stride`.
    pub fn new(series: Series<T, N>, default: Array<T, N>) -> Self {
        Self { series, default }
    }
}

impl<T: Scalar, const N: usize> Source for ArraySource<T, N> {
    type Instant = Instant;
    type Payload = Array<T, N>;
    type Pass = ArrayPass<T, N>;

    fn total_num_events(&self) -> Option<usize> {
        Some(self.series.len())
    }

    fn initial(&self) -> Array<T, N> {
        self.default.clone()
    }

    fn init(
        &self,
    ) -> (
        impl Stream<Item = Event<Instant, Array<T, N>>> + Send + 'static,
        impl FnMut(Instant, Array<T, N>, &mut Array<T, N>) -> usize + Send + 'static,
    ) {
        let (hist_tx, hist_rx) = mpsc::channel(64);

        // Clone the series for the spawned driver; the spec stays
        // borrowable so the same source can drive multiple sessions.
        let series = self.series.clone();
        tokio::spawn(async move {
            let view = series.view();
            for i in 0..view.len() {
                let arr = view.elem(i).to_array();
                if hist_tx.send((view.timestamp(i), arr)).await.is_err() {
                    break;
                }
            }
        });

        (
            receiver_stream(hist_rx),
            |_, payload, output: &mut Array<T, N>| {
                output.assign(payload.view());
                1
            },
        )
    }
}

/// A historical-only source replaying `series`, holding `default` before the
/// first event.
pub fn array_source<T: Scalar, const N: usize>(
    series: Series<T, N>,
    default: Array<T, N>,
) -> ArraySource<T, N> {
    ArraySource::new(series, default)
}

// ===========================================================================
// IterSource — driven by a factory of (timestamp, value) iterators.
// ===========================================================================

/// Boxed iterator type produced by an [`IterSource`] factory.
type EventIter<T> = Box<dyn Iterator<Item = (Instant, T)> + Send>;

/// Shared, cheaply-cloned factory that produces fresh event iterators.
///
/// Stored behind `Arc<...>` so the [`IterSource`] spec satisfies `Clone`
/// (refcount bump only) and can drive multiple scenario sessions, each with a
/// freshly built iterator — and `Send`, as [`Source`] requires (the spec
/// moves into the session's lazily-started feed).
type IterFactory<T> = Arc<dyn Fn() -> EventIter<T> + Send + Sync>;

/// A source driven by a factory of `(timestamp, event)` iterators.
///
/// More flexible than [`ArraySource`] - supports lazy
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
    total_num_events: Option<usize>,
}

impl<T: Clone + Send + 'static> Clone for IterSource<T> {
    fn clone(&self) -> Self {
        Self {
            factory: Arc::clone(&self.factory),
            default: self.default.clone(),
            total_num_events: self.total_num_events,
        }
    }
}

impl<T: Clone + Send + 'static> IterSource<T> {
    /// Create from an iterator factory and a default output value.
    ///
    /// `factory` is called once per [`Source::init`] invocation and must
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
            total_num_events: None,
        }
    }

    /// Create from a fixed `Vec` of `(timestamp, value)` events.
    ///
    /// The vector is shared across clones via `Arc` and cloned on each
    /// [`Source::init`] call to produce a fresh iterator.  The estimated
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
        .with_total_num_events(count)
    }

    /// Advertise an estimated total event count.
    ///
    /// Call this when the iterator length is known at construction time
    /// (e.g. clock sources backed by a `Vec`).  Used only for progress
    /// reporting ([`Session::total_num_events`](crate::Session::total_num_events)).
    pub fn with_total_num_events(mut self, count: usize) -> Self {
        self.total_num_events = Some(count);
        self
    }
}

impl<T: Clone + Send + Sync + 'static> Source for IterSource<T> {
    type Instant = Instant;
    type Payload = T;
    type Pass = Ref<T>;

    fn total_num_events(&self) -> Option<usize> {
        self.total_num_events
    }

    fn initial(&self) -> T {
        self.default.clone()
    }

    fn init(
        &self,
    ) -> (
        impl Stream<Item = Event<Instant, T>> + Send + 'static,
        impl FnMut(Instant, T, &mut T) -> usize + Send + 'static,
    ) {
        let (hist_tx, hist_rx) = mpsc::channel(64);

        let iter = (self.factory)();
        tokio::spawn(async move {
            for (ts, value) in iter {
                if hist_tx.send((ts, value)).await.is_err() {
                    break;
                }
            }
        });

        (receiver_stream(hist_rx), |_, payload, output: &mut T| {
            *output = payload;
            1
        })
    }
}

/// A source driven by `factory`, holding `default` before the first event.
/// `factory` is called once per session to produce a fresh
/// `(timestamp, value)` iterator. Chain
/// [`with_total_num_events`](IterSource::with_total_num_events) when the length
/// is known (progress reporting only).
pub fn iter_source<I, F, T>(factory: F, default: T) -> IterSource<T>
where
    T: Clone + Send + 'static,
    I: Iterator<Item = (Instant, T)> + Send + 'static,
    F: Fn() -> I + Send + Sync + 'static,
{
    IterSource::new(factory, default)
}

/// A source replaying a fixed `Vec` of `(timestamp, value)` events, holding
/// `T::default()` before the first — the [`iter_source`] shorthand for data
/// already in memory. Use
/// [`IterSource::from_vec_with_default`] to set the pre-first-event value.
pub fn vec_source<T: Clone + Default + Send + Sync + 'static>(
    events: Vec<(Instant, T)>,
) -> IterSource<T> {
    IterSource::from_vec(events)
}

// ===========================================================================
// Pulse — bare () triggers for the clock-gated operators.
// ===========================================================================

/// A pulse (bare-trigger) source: emits a `()` at each of the given
/// timestamps, passed **by value** (`Val<()>`) so it wires directly into the
/// clock-gated operators' `UnitPort` inputs. `Clone` and reusable across
/// sessions; its events drain through a spawned tokio task like the other
/// sources.
#[derive(Clone)]
pub struct PulseSource {
    timestamps: Arc<Vec<Instant>>,
}

impl Source for PulseSource {
    type Instant = Instant;
    type Payload = ();
    type Pass = Val<()>;

    fn total_num_events(&self) -> Option<usize> {
        Some(self.timestamps.len())
    }

    fn initial(&self) {}

    fn init(
        &self,
    ) -> (
        impl Stream<Item = Event<Instant, ()>> + Send + 'static,
        impl FnMut(Instant, (), &mut ()) -> usize + Send + 'static,
    ) {
        let (hist_tx, hist_rx) = mpsc::channel(64);

        let timestamps = self.timestamps.clone();
        tokio::spawn(async move {
            for ts in timestamps.iter().copied() {
                if hist_tx.send((ts, ())).await.is_err() {
                    break;
                }
            }
        });

        (receiver_stream(hist_rx), |_, _payload, _output: &mut ()| 1)
    }
}

/// Create a pulse source from explicit timestamps.
///
/// The output node holds `()` (zero-sized, purely a trigger) passed by value.
/// The resulting [`PulseSource`] is `Clone` and reusable across multiple
/// scenario sessions.
pub fn pulse(timestamps: Vec<Instant>) -> PulseSource {
    PulseSource {
        timestamps: Arc::new(timestamps),
    }
}
