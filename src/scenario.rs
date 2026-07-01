//! The engine: graph builder + event-loop driver.
//!
//! [`Scenario`] builds a `flowgraph` graph whose source cells are `push_source`
//! nodes and registers a [`Source`](crate::source::Source) per source cell.
//! [`Session`] owns the built graph + a worker [`Pool`] + the shared
//! [`Clock`](crate::operators::Clock), and drives them via [`Session::run`].
//!
//! The event loop is [`flowgraph::ingest`]: a timestamp-ordered watermark merge
//! with same-timestamp coalescing. Each source becomes a [`Feed`] whose
//! `watermark`/`take_through` **block** on the source's historical channel to
//! reveal the next in-order row; the sync driver runs on a `spawn_blocking`
//! thread while the sources' async producer tasks stream rows into the channels
//! with back-pressure. Per coalesced batch the driver pokes each fired source's
//! cell (via the typed `Graph::state_mut`, which marks the dirty cone) through
//! the source's [`write`](Source::write), advances the [`Clock`] to the batch
//! timestamp, and runs one `stabilize`.
//!
//! Time is threaded out-of-band through the [`Clock`]: the apply step sets it to
//! the batch timestamp before `stabilize`, and time-reading operators (e.g.
//! [`Record`]) read it during compute.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

use flowgraph::core::Pool;
use flowgraph::ingest::{Driver, Feed, Progress};
use flowgraph::typed::{
    Graph, GraphBuilder, Handle, HandlesInterface, InterfaceHandles, RefPort, RefSource,
    RefViewPort, Segment, SourceHandle, ViewPort,
};

use crate::operators::{ArrayValue, Clock, Record};
use crate::source::Source;
use crate::{Array, Instant, Scalar, Series};

/// Shared shutdown flag for cooperative cancellation of the event loop.
pub type ShutdownFlag = Arc<AtomicBool>;

// ---------------------------------------------------------------------------
// SourceFeed — bridges a source's historical channel to an ingest `Feed`
// ---------------------------------------------------------------------------

/// A [`Feed`] over a source's historical event channel. `watermark`/`take_through`
/// **block** (`blocking_recv`) on the channel until the next in-order row arrives
/// or the channel closes (= the source is exhausted) — the driver runs on a
/// dedicated blocking thread, so this "wait for the next row before advancing"
/// is exactly the historical-ordering constraint. A one-slot peek buffer keeps
/// `watermark` idempotent before a drain.
struct SourceFeed<E> {
    rx: mpsc::Receiver<(Instant, E)>,
    buf: Option<(Instant, E)>,
    closed: bool,
}

impl<E: Send + 'static> SourceFeed<E> {
    fn new(rx: mpsc::Receiver<(Instant, E)>) -> Self {
        Self { rx, buf: None, closed: false }
    }

    /// Block until the next row is buffered (or the channel closes); return its
    /// timestamp.
    fn peek_ts(&mut self) -> Option<Instant> {
        if self.buf.is_none() && !self.closed {
            match self.rx.blocking_recv() {
                Some(item) => self.buf = Some(item),
                None => self.closed = true,
            }
        }
        self.buf.as_ref().map(|(t, _)| *t)
    }
}

impl<E: Send + 'static> Feed for SourceFeed<E> {
    type Time = Instant;
    type Delta = Vec<E>;

    fn watermark(&mut self) -> Option<Instant> {
        self.peek_ts()
    }

    fn take_through(&mut self, through: Instant) -> Option<Vec<E>> {
        let mut batch = Vec::new();
        while let Some(t) = self.peek_ts() {
            if t > through {
                break;
            }
            batch.push(self.buf.take().unwrap().1);
        }
        (!batch.is_empty()).then_some(batch)
    }
}

// ---------------------------------------------------------------------------
// Source erasure — a per-source registrar (async init) → feed-adder (driver)
// ---------------------------------------------------------------------------

/// Registers one source's feed into the driver. Created on the driver thread
/// (so nothing here needs the tokio runtime), capturing the source's channel +
/// write state + cell handle + clock — all `Send`, so it crosses to the driver
/// thread. The `Rc` counter (driver-thread-local) accumulates the run's logical
/// event count for the progress callback.
type FeedAdder = Box<dyn FnOnce(&mut Driver<Instant>, Rc<Cell<usize>>) + Send>;

/// Initialises one source (on the async side, spawning its producer task) and
/// returns a [`FeedAdder`]. Kept in the [`Scenario`]/[`Session`] until
/// [`run`](Session::run), which is when a source's producers should start.
type Registrar = Box<dyn FnOnce() -> FeedAdder>;

// ---------------------------------------------------------------------------
// Scenario — builder
// ---------------------------------------------------------------------------

/// Builds a `flowgraph` graph plus its source registrations.
pub struct Scenario {
    builder: GraphBuilder,
    clock: Clock,
    registrars: Vec<Registrar>,
    /// Sum of every source's [`Source::estimated_event_count`]; `None` once any
    /// source can't estimate. Used only for the [`Session::run`] progress callback.
    estimated_total: Option<usize>,
}

impl Scenario {
    pub fn new() -> Self {
        Self {
            builder: GraphBuilder::new(),
            clock: Clock::new(),
            registrars: Vec::new(),
            estimated_total: Some(0),
        }
    }

    /// Register a constant node (an externally-set source cell with no feed).
    /// Mutate it via [`Session::cell_mut`] + [`Session::flush`]; wire it
    /// downstream via the deref'd plain handle (`*h`).
    pub fn add_const<T: Send + Sync + 'static>(&mut self, value: T) -> SourceHandle<RefSource<T>> {
        self.builder.push_source(RefSource::new(value))
    }

    /// Register a [`Segment`] (most operators implement
    /// [`flowgraph::typed::Operator`], which is one). `inputs` are the upstream
    /// handles, shaped like the segment's input tree.
    pub fn add_operator<O, H>(
        &mut self,
        operator: O,
        inputs: H,
    ) -> <O::Outputs as InterfaceHandles>::HandlesOwned
    where
        H: HandlesInterface,
        O: Segment<Inputs = H::Interface>,
        O::Outputs: InterfaceHandles,
    {
        self.builder.push(operator, inputs)
    }

    /// Bridge a whole-array reference handle (a source cell / [`add_const`]
    /// (Self::add_const) / any `RefPort<Array<T, N>>`) into the
    /// `ViewPort<ArrayValue<T, N>>` view currency the operators speak, via a
    /// zero-copy [`AsView`](crate::operators::AsView). Source outputs are
    /// whole-array references; everything downstream is views.
    pub fn as_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<RefPort<Array<T, N>>>,
    ) -> Handle<ViewPort<ArrayValue<T, N>>> {
        self.builder.push(crate::operators::AsView::<T, N>::new(), data)
    }

    /// Bridge a by-value view handle into the by-reference `RefViewPort` currency
    /// the carry-join combines ([`Stack`](crate::operators::Stack) /
    /// [`Concat`](crate::operators::Concat)) consume, via
    /// [`RefArrayView`](crate::operators::RefArrayView). Lets independently-produced
    /// view handles (e.g. separate feature columns) be collected into the
    /// `&[RefViewPort]` slice those combines take.
    pub fn ref_array_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<ViewPort<ArrayValue<T, N>>>,
    ) -> Handle<RefViewPort<ArrayValue<T, N>>> {
        self.builder.push(crate::operators::RefArrayView::<T, N>::new(), data)
    }

    /// Re-derive a by-reference view handle (e.g. a
    /// [`Split`](crate::operators::Split) row) as the by-value `ViewPort` currency
    /// the elementwise operators consume, via
    /// [`DerefArrayView`](crate::operators::DerefArrayView). The inverse of
    /// [`ref_array_view`](Self::ref_array_view).
    pub fn deref_array_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<RefViewPort<ArrayValue<T, N>>>,
    ) -> Handle<ViewPort<ArrayValue<T, N>>> {
        self.builder.push(crate::operators::DerefArrayView::<T, N>::new(), data)
    }

    /// Materialize a by-value view handle back into an **owned** whole-array
    /// `RefPort<Array<T, N>>` cell, via [`Own`](crate::operators::Own) — the
    /// inverse of [`as_view`](Self::as_view), e.g. before a whole-array consumer
    /// such as a `PyClassOperator`.
    pub fn own<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<ViewPort<ArrayValue<T, N>>>,
    ) -> Handle<RefPort<Array<T, N>>> {
        self.builder.push(crate::operators::Own::<T, N>::new(), data)
    }

    /// Register a [`Record`] for a data stream, wiring the driver's [`Clock`]
    /// so each recorded row is stamped with the current event time. `Record`
    /// needs the clock, so it cannot go through [`add_operator`](Self::add_operator).
    pub fn add_record<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<ViewPort<ArrayValue<T, N>>>,
    ) -> Handle<RefPort<Series<T>>> {
        self.builder.push(Record::new(self.clock.clone()), data)
    }

    /// Register a [`Record`] whose recorded `Series` keeps only the history
    /// within `retention`, dropping older rows to bound memory. Use for records
    /// consumed only within a fixed look-back window (rolling / lag operators, a
    /// predictor's training tail); the bound must cover every consumer's deepest
    /// look-back. Otherwise identical to [`add_record`](Self::add_record).
    pub fn add_record_retained<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<ViewPort<ArrayValue<T, N>>>,
        retention: crate::Retention,
    ) -> Handle<RefPort<Series<T>>> {
        self.builder
            .push(Record::with_retention(self.clock.clone(), retention), data)
    }

    /// The driver's [`Clock`]. For building custom time-reading operators (held
    /// in their own state, like [`Record`]); ordinary operators are clock-free.
    pub fn clock(&self) -> Clock {
        self.clock.clone()
    }

    /// Register a Python operator in **return mode** (feature `python`).
    /// `source` is a Python expression evaluating to a callable that takes
    /// `inputs.len()` `float64` ndarrays and returns a 1-D `float64` ndarray of
    /// length `out_len`, e.g. `"lambda a, b: a + b"`. Operators run on the shared
    /// (free-threaded) interpreter and execute in parallel on the pool. See the
    /// `operators::python` module docs for the data model, retention contract, and setup.
    #[cfg(feature = "python")]
    pub fn add_py_operator(
        &mut self,
        source: &str,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_len: usize,
    ) -> Handle<RefPort<Array<f64, 1>>> {
        self.builder
            .push(crate::operators::PyOperator::<1, 1>::new(source, out_len), inputs)
    }

    /// Register a Python operator in **write mode** (feature `python`, zero-copy
    /// output). `source` evaluates to a callable `f(out, *inputs)` taking a
    /// writable `float64` ndarray `out` (aliasing the output buffer) followed by
    /// the input ndarrays; it writes its `out_len` results into `out` in place
    /// and its return value is ignored, e.g. `"lambda out, a, b: np.add(a, b, out=out)"`.
    /// The `out` view must not escape the call (see the `operators::python` module docs).
    #[cfg(feature = "python")]
    pub fn add_py_operator_writing(
        &mut self,
        source: &str,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_len: usize,
    ) -> Handle<RefPort<Array<f64, 1>>> {
        self.builder.push(
            crate::operators::PyOperator::<1, 1>::writing(source, out_len),
            inputs,
        )
    }

    /// Register a class-based Python operator (feature `python`). `source` is a
    /// Python program binding the operator instance to `__op__`, implementing the
    /// legacy contract `init(inputs, timestamp) -> state` and
    /// `compute(state, inputs, output, timestamp, produced) -> bool`. The driver
    /// [`Clock`] is wired through so the operator sees event time. `out_shape` is
    /// the output element shape (`[]` for a scalar). See [`PyClassOperator`](crate::operators::PyClassOperator).
    #[cfg(feature = "python")]
    pub fn add_py_class_operator(
        &mut self,
        source: &str,
        params: crate::operators::PyParams,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_shape: &[usize],
    ) -> Handle<RefPort<Array<f64, 1>>> {
        self.builder.push(
            crate::operators::PyClassOperator::<flowgraph::typed::RefPorts<Array<f64, 1>>, 1>::from_source(
                source,
                params,
                out_shape.to_vec(),
                self.clock.clone(),
            ),
            inputs,
        )
    }

    /// Register a class-based Python operator loaded from a plain `.py` file
    /// (feature `python`). The file defines `build(**kwargs)` (called with
    /// `params`) or binds `__op__`; see [`PyClassOperator`](crate::operators::PyClassOperator). Convenience for
    /// all-`Array<f64, 1>` inputs; heterogeneous (Array + Series) operators register
    /// via [`add_operator`](Self::add_operator) with
    /// `PyClassOperator::<I, NO>::from_file(..)`.
    #[cfg(feature = "python")]
    pub fn add_py_operator_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
        params: crate::operators::PyParams,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_shape: &[usize],
    ) -> Handle<RefPort<Array<f64, 1>>> {
        self.builder.push(
            crate::operators::PyClassOperator::<flowgraph::typed::RefPorts<Array<f64, 1>>, 1>::from_file(
                path,
                params,
                out_shape.to_vec(),
                self.clock.clone(),
            ),
            inputs,
        )
    }

    /// Register a [`Source`]. Its output cell is a `RefSource` node; the source's
    /// producer task is spawned at [`run`](Session::run), and its events are
    /// merged in timestamp order and written into the cell via [`Source::write`].
    pub fn add_source<S: Source>(
        &mut self,
        source: S,
        initial: S::Output,
    ) -> Handle<RefPort<S::Output>>
    where
        S::Output: Send + Sync + 'static,
    {
        let handle = self.builder.push_source(RefSource::new(initial));
        // Accumulate the progress estimate before the source is moved into the
        // registrar; a single un-estimable source makes the whole total unknown.
        let est = source.estimated_event_count();
        self.estimated_total = match (self.estimated_total, est) {
            (Some(acc), Some(n)) => Some(acc.saturating_add(n)),
            _ => None,
        };
        let clock = self.clock.clone();
        self.registrars.push(Box::new(move || -> FeedAdder {
            // On the async side: spawn the source's producer, take its event
            // channel + write state. (The `init` output is unused — the graph's
            // `push_source` node already holds the initial value.)
            let (hist_rx, _output, state) = source.init(Instant::MIN);
            Box::new(move |driver: &mut Driver<Instant>, counter: Rc<Cell<usize>>| {
                let mut write_state = state;
                driver.add_feed(SourceFeed::new(hist_rx), move |g, ts, events: Vec<S::Event>| {
                    clock.set(ts);
                    let out = g.state_mut(handle);
                    let mut n = 0usize;
                    for ev in events {
                        n += S::write(&mut write_state, ev, out, ts);
                    }
                    counter.set(counter.get() + n);
                });
            })
        }));
        *handle
    }

    /// Finalize the graph, producing a runnable [`Session`] with a single-threaded pool.
    pub fn build(self) -> Session {
        self.build_with_threads(0)
    }

    /// Like [`build`](Self::build) but with `n` extra worker threads in the pool.
    pub fn build_with_threads(self, n: usize) -> Session {
        let Scenario {
            builder,
            clock,
            registrars,
            estimated_total,
        } = self;
        Session {
            graph: Some(Graph::from_builder(builder)),
            pool: Some(Pool::new(n)),
            clock,
            registrars,
            estimated_total,
        }
    }
}

impl Default for Scenario {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Session — runtime + event loop
// ---------------------------------------------------------------------------

/// A live execution: owns the graph, the worker pool, and the clock. The graph
/// and pool are `Option`-held so [`run`](Self::run) can lend them to the driver
/// thread and reclaim them (they are `Some` except during a `run`).
pub struct Session {
    graph: Option<Graph>,
    pool: Option<Pool>,
    clock: Clock,
    registrars: Vec<Registrar>,
    estimated_total: Option<usize>,
}

// SAFETY: the graph and pool are `Send`; the clock is an `Arc<AtomicI64>`. The
// registrars capture the (possibly `!Send`) source specs but are only ever run
// on the async task that owns the session, before their `Send` feed-adders (and
// only those) cross to the driver thread. The session is driven from a single
// task.
unsafe impl Send for Session {}

impl Session {
    /// Sum of every source's estimated event count, or `None` if any source
    /// couldn't estimate. Pair it with [`run`](Self::run)'s `on_flush` count to
    /// drive a progress bar. Advisory.
    pub fn estimated_event_count(&self) -> Option<usize> {
        self.estimated_total
    }

    /// Immutable access to a by-reference node value.
    pub fn value<T: Send + Sync + 'static>(&self, handle: Handle<RefPort<T>>) -> &T {
        self.graph.as_ref().unwrap().ref_view(handle)
    }

    /// Mutable access to a source cell's value (marks its dirty cone). Only
    /// source cells ([`Scenario::add_const`] / [`Scenario::add_source`]) can be
    /// poked; operator outputs are state-owned and engine-written.
    pub fn cell_mut<T: Send + Sync + 'static>(
        &mut self,
        handle: SourceHandle<RefSource<T>>,
    ) -> &mut T {
        self.graph.as_mut().unwrap().state_mut(handle)
    }

    /// Manually advance the clock and stabilize (the synchronous-step API).
    /// Call after mutating one or more source cells via [`cell_mut`](Self::cell_mut).
    pub fn flush(&mut self, timestamp: Instant) {
        self.clock.set(timestamp);
        let pool = self.pool.as_mut().unwrap();
        self.graph.as_mut().unwrap().stabilize(pool);
    }

    /// Drive the event loop until every source is exhausted.
    ///
    /// `on_flush(batch_ts, events_processed_so_far)` is invoked once per
    /// coalesced batch.
    pub async fn run(&mut self, on_flush: impl FnMut(Instant, usize) + Send + 'static) {
        self.run_with_shutdown(on_flush, Arc::new(AtomicBool::new(false)))
            .await;
    }

    /// Like [`run`](Self::run), but cooperatively cancellable via `shutdown`.
    pub async fn run_with_shutdown(
        &mut self,
        on_flush: impl FnMut(Instant, usize) + Send + 'static,
        shutdown: ShutdownFlag,
    ) {
        if self.registrars.is_empty() {
            return;
        }
        // Initialise every source on the async side (spawns its producer task),
        // producing `Send` feed-adders that cross to the driver thread.
        let adders: Vec<FeedAdder> = self.registrars.drain(..).map(|r| r()).collect();
        let graph = self.graph.take().unwrap();
        let pool = self.pool.take().unwrap();
        let clock = self.clock.clone();

        // Run the synchronous merge driver on a blocking thread: its feeds block
        // on the source channels, which the producer tasks fill concurrently.
        let (graph, pool) = tokio::task::spawn_blocking(move || {
            let counter = Rc::new(Cell::new(0usize));
            let mut driver = Driver::new(graph, pool);
            for adder in adders {
                adder(&mut driver, Rc::clone(&counter));
            }
            let mut on_flush = on_flush;
            loop {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                // `on_flush` runs after stabilize with the batch timestamp (from
                // the clock the apply step set) and the cumulative event count.
                let progress = driver.step(|_g| on_flush(clock.get(), counter.get()));
                match progress {
                    // Historical feeds always make progress until exhausted, so
                    // `Idle` should not occur; treat it as a safe stop.
                    Progress::Done | Progress::Idle => break,
                    Progress::Stabilized => {}
                }
            }
            driver.into_parts()
        })
        .await
        .unwrap();

        self.graph = Some(graph);
        self.pool = Some(pool);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{Filter, add};
    use crate::sources::ArraySource;
    use crate::{Array, ArrayView, Series};

    fn tss(xs: &[i64]) -> Vec<Instant> {
        xs.iter().copied().map(Instant::from_nanos).collect()
    }

    fn src(ts: &[i64], vals: &[f64]) -> ArraySource<f64, 0> {
        ArraySource::new(Series::from_vec(&[], tss(ts), vals.to_vec()), Array::scalar(0.0))
    }

    /// Replay [10,20,30] @ [1,2,3] into a Record.
    #[tokio::test]
    async fn run_single_source_record() {
        let mut sc = Scenario::new();
        let h = sc.add_source(src(&[1, 2, 3], &[10.0, 20.0, 30.0]), Array::scalar(0.0));
        let hv = sc.as_view(h);
        let hrec = sc.add_record(hv);

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(s.values(), &[10.0, 20.0, 30.0]);
    }

    /// Staggered timestamps; the un-fired input keeps its stale value.
    /// ts1:10+0, ts2:10+20, ts3:30+40 → [10,30,70].
    #[tokio::test]
    async fn run_two_sources_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(src(&[1, 3], &[10.0, 30.0]), Array::scalar(0.0));
        let hb = sc.add_source(src(&[2, 3], &[20.0, 40.0]), Array::scalar(0.0));
        let (hav, hbv) = (sc.as_view(ha), sc.as_view(hb));
        let ho = sc.add_operator(add::<f64, 0>(), (hav, hbv));
        let hrec = sc.add_record(ho);

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(s.values(), &[10.0, 30.0, 70.0]);
    }

    /// Two sources at the same timestamps → one coalesced batch each.
    /// ts1:10+100, ts2:20+200 → [110,220].
    #[tokio::test]
    async fn run_coalescing() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(src(&[1, 2], &[10.0, 20.0]), Array::scalar(0.0));
        let hb = sc.add_source(src(&[1, 2], &[100.0, 200.0]), Array::scalar(0.0));
        let (hav, hbv) = (sc.as_view(ha), sc.as_view(hb));
        let ho = sc.add_operator(add::<f64, 0>(), (hav, hbv));
        let hrec = sc.add_record(ho);

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.timestamps(), tss(&[1, 2]).as_slice());
        assert_eq!(s.values(), &[110.0, 220.0]);
    }

    /// The cutoff must survive the driver — dropped ticks produce no Record row.
    /// [1,5,2,10] keep >3 → (2,5),(4,10).
    #[tokio::test]
    async fn run_filter_cutoff() {
        let mut sc = Scenario::new();
        let h = sc.add_source(src(&[1, 2, 3, 4], &[1.0, 5.0, 2.0, 10.0]), Array::scalar(0.0));
        let hv = sc.as_view(h);
        let hf = sc.add_operator(Filter::<_, 0>(|v: ArrayView<f64, 0>| v.to_contiguous()[0] > 3.0), hv);
        let hrec = sc.add_record(hf);

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.len(), 2);
        assert_eq!(s.timestamps(), tss(&[2, 4]).as_slice());
        assert_eq!(s.values(), &[5.0, 10.0]);
    }

    /// `on_flush` fires once per coalesced batch (3 distinct timestamps here).
    /// The callback runs on the driver's `spawn_blocking` thread, so results are
    /// collected through a `Send` shared vec.
    #[tokio::test]
    async fn on_flush_per_batch() {
        use std::sync::{Arc, Mutex};

        let mut sc = Scenario::new();
        let h = sc.add_source(src(&[1, 2, 3], &[10.0, 20.0, 30.0]), Array::scalar(0.0));
        let hv = sc.as_view(h);
        let _ = sc.add_record(hv);

        let mut session = sc.build();
        let batches = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&batches);
        session.run(move |ts, _| sink.lock().unwrap().push(ts.as_nanos())).await;
        assert_eq!(*batches.lock().unwrap(), vec![1, 2, 3]);
    }
}
