//! The engine: graph builder + async event-loop driver.
//!
//! [`Scenario`] builds a `flowgraph` graph whose source cells are
//! `push_source` nodes, and registers a [`Source`](crate::source::Source) per
//! source cell. [`Session`] owns the built graph + a worker [`Pool`] + the
//! shared [`Clock`](crate::operators::Clock) + the per-source channel state, and
//! drives them via [`Session::run`].
//!
//! The event loop is a timestamp merge-heap with same-timestamp coalescing, a
//! historical-ordering constraint, and live-event clamping. The per-batch apply
//! step writes each fired source's event into its node state via the typed
//! `Graph::state_mut(SourceHandle<T>)` (which marks the dirty cone) through a
//! per-source monomorphized write function, advances the [`Clock`](crate::operators::Clock)
//! to the batch timestamp, and runs one `stabilize`.

use std::any::TypeId;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::future::poll_fn;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use futures::stream::{FuturesUnordered, StreamExt};

use flowgraph::core::Pool;
use flowgraph::typed::{
    Graph, GraphBuilder, Handle, InterfaceHandles, RefPort, RefSource, Segment, SourceHandle,
};

use crate::operators::{Clock, Record};
use crate::source::{PollFn, Source};
use crate::{Array, Instant, PeekableReceiver, Scalar, Series};

/// Shared shutdown flag for cooperative cancellation of the event loop.
pub type ShutdownFlag = Arc<AtomicBool>;

// ---------------------------------------------------------------------------
// Flow-side source erasure — typed graph writes through `state_mut`
// ---------------------------------------------------------------------------

/// Type-erased graph-write function pointer for a source, monomorphized per
/// source type by [`erased_flow_source`].
///
/// # Parameters
///
/// * `state_ptr: *mut u8` — points to `S::State`.
/// * `rx_ptr: *mut u8` — points to [`PeekableReceiver<(Instant, S::Event)>`].
/// * `graph: &mut Graph` — the typed graph (the write marks the dirty cone).
/// * `index: usize` — the source node's slot index (`SourceHandle::index`).
/// * `timestamp: Instant` — coalesced batch timestamp.
///
/// # Returns
///
/// The number of logical events the consumed channel item represented (a
/// batched source returns its batch size); `0` if nothing was buffered.
type FlowWriteFn = unsafe fn(*mut u8, *mut u8, &mut Graph, usize, Instant) -> usize;

/// Type-erased representation of a flow source: fresh channel receivers + write
/// state from `init_fn`, plus the monomorphized poll/write/drop functions.
struct ErasedFlowSource {
    estimated_event_count: Option<usize>,
    /// Returns `(hist_rx_ptr, live_rx_ptr, state_ptr)`, each from
    /// [`Box::into_raw`]. The source's `init` output value is dropped — the
    /// graph's `push_source` node owns the live value.
    #[allow(clippy::type_complexity)]
    init_fn: Box<dyn FnOnce(Instant) -> (*mut u8, *mut u8, *mut u8)>,
    poll_fn: PollFn,
    write_fn: FlowWriteFn,
    rx_drop_fn: unsafe fn(*mut u8),
    state_drop_fn: unsafe fn(*mut u8),
}

fn erased_flow_source<S: Source>(source: S) -> ErasedFlowSource
where
    S::Output: Send + Sync + 'static,
{
    ErasedFlowSource {
        estimated_event_count: source.estimated_event_count(),
        init_fn: Box::new(move |timestamp: Instant| {
            let (hist, live, output, state) = source.init(timestamp);
            // The flowgraph node already holds the `push_source(RefSource::new(initial))`
            // value, so the source's freshly-allocated init output is unused.
            drop(output);
            (
                Box::into_raw(Box::new(PeekableReceiver::new(hist))) as *mut u8,
                Box::into_raw(Box::new(PeekableReceiver::new(live))) as *mut u8,
                Box::into_raw(Box::new(state)) as *mut u8,
            )
        }),
        poll_fn: flow_poll_fn::<S>,
        write_fn: flow_write_fn::<S>,
        rx_drop_fn: flow_drop_fn::<PeekableReceiver<(Instant, S::Event)>>,
        state_drop_fn: flow_drop_fn::<S::State>,
    }
}

/// Type-erased poll function, monomorphised per event type.
unsafe fn flow_poll_fn<S: Source>(
    rx_ptr: *mut u8,
    ctx: &mut Context<'_>,
) -> Poll<Option<Instant>> {
    let rx = unsafe { &mut *(rx_ptr as *mut PeekableReceiver<(Instant, S::Event)>) };
    rx.poll_pending(ctx).map(|opt| opt.map(|item| item.0))
}

/// The typed layer's per-node state bundling for a `RefSource<T>` node: input
/// shape + engine-stored output payload tree (`(bool, &T)` at `'static`) + the
/// user state (`T` itself for a source). Mirrors `NodeState` in
/// `flowgraph/src/typed/graph.rs`; the `TypeId` assert in [`flow_write_fn`]
/// re-checks the layout on every write (same check the typed
/// `Graph::state_mut` performs), so a future layout change fails loud.
type SourceNodeState<T> = (Box<[usize]>, MaybeUninit<(bool, &'static T)>, T);

/// Type-erased graph-write function, monomorphised per source type. Writes the
/// buffered event into the source node's state cell via the slot-indexed
/// untyped layer (`Graph::inner_mut().state_mut(index)`, which marks the dirty
/// cone), re-typing the [`ErasedCell`](flowgraph::core::ErasedCell) payload.
unsafe fn flow_write_fn<S: Source>(
    state_ptr: *mut u8,
    rx_ptr: *mut u8,
    graph: &mut Graph,
    index: usize,
    timestamp: Instant,
) -> usize
where
    S::Output: Send + Sync + 'static,
{
    let state = unsafe { &mut *(state_ptr as *mut S::State) };
    let rx = unsafe { &mut *(rx_ptr as *mut PeekableReceiver<(Instant, S::Event)>) };
    let Some(item) = rx.take_pending() else {
        return 0;
    };
    let cell = graph.inner_mut().state_mut(index);
    assert!(
        cell.type_id() == TypeId::of::<SourceNodeState<S::Output>>(),
        "source slot state type mismatch"
    );
    let (_, _, output) = unsafe { &mut *cell.get().cast::<SourceNodeState<S::Output>>() };
    S::write(state, item.1, output, timestamp)
}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn flow_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}

// ---------------------------------------------------------------------------
// Channels / merge entries / receive future (mirrors scenario::queue)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
enum ChannelKind {
    Hist,
    Live,
}

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
struct HeapEntry {
    ts: Instant,
    source_idx: usize,
    kind: ChannelKind,
}

/// One pending channel receive; resolves to `(source_idx, kind, Option<ts>)`.
/// Holds a raw `rx_ptr` so it does not borrow the owning [`SourceSlot`]; must
/// not outlive the [`Session`] (whose `slots` own the receivers).
struct ErasedRecvFuture {
    rx_ptr: *mut u8,
    poll_fn: PollFn,
    source_idx: usize,
    kind: ChannelKind,
}

// SAFETY: rx_ptr points to a `PeekableReceiver<E>`, which is `Send`.
unsafe impl Send for ErasedRecvFuture {}

impl std::future::Future for ErasedRecvFuture {
    type Output = (usize, ChannelKind, Option<Instant>);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let source_idx = self.source_idx;
        let kind = self.kind;
        unsafe { (self.poll_fn)(self.rx_ptr, cx) }.map(|opt_ts| (source_idx, kind, opt_ts))
    }
}

// ---------------------------------------------------------------------------
// SourceSlot — per-source channel state pointing at a flowgraph source node
// ---------------------------------------------------------------------------

struct SourceSlot {
    /// Slot index of this source's `push_source` node (`SourceHandle::index`);
    /// `write_fn` re-types the slot's state cell payload.
    index: usize,
    hist_rx_ptr: *mut u8,
    live_rx_ptr: *mut u8,
    /// Per-source write state (`Source::State`), threaded into every `write_fn`.
    state_ptr: *mut u8,
    poll_fn: PollFn,
    write_fn: FlowWriteFn,
    rx_drop_fn: unsafe fn(*mut u8),
    state_drop_fn: unsafe fn(*mut u8),
}

// SAFETY: the slot owns the two receiver allocations, which are `Send`.
unsafe impl Send for SourceSlot {}

impl SourceSlot {
    fn rx_ptr(&self, kind: ChannelKind) -> *mut u8 {
        match kind {
            ChannelKind::Hist => self.hist_rx_ptr,
            ChannelKind::Live => self.live_rx_ptr,
        }
    }

    /// # Safety
    /// The returned future must not outlive this slot.
    unsafe fn recv_future(&self, source_idx: usize, kind: ChannelKind) -> ErasedRecvFuture {
        ErasedRecvFuture {
            rx_ptr: self.rx_ptr(kind),
            poll_fn: self.poll_fn,
            source_idx,
            kind,
        }
    }
}

impl Drop for SourceSlot {
    fn drop(&mut self) {
        unsafe {
            (self.rx_drop_fn)(self.hist_rx_ptr);
            (self.rx_drop_fn)(self.live_rx_ptr);
            (self.state_drop_fn)(self.state_ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario — builder
// ---------------------------------------------------------------------------

struct SourceDescriptor {
    /// Slot index of the source's `push_source` node.
    index: usize,
    erased: ErasedFlowSource,
}

/// Builds a `flow` graph plus its source registrations.
pub struct Scenario {
    builder: GraphBuilder,
    clock: Clock,
    sources: Vec<SourceDescriptor>,
    /// Sum of every source's [`Source::estimated_event_count`]; `None` once any
    /// source can't estimate (then the whole total is unknown). Used only for
    /// the [`Session::run`] progress callback.
    estimated_total: Option<usize>,
}

impl Scenario {
    pub fn new() -> Self {
        Self {
            builder: GraphBuilder::new(),
            clock: Clock::new(),
            sources: Vec::new(),
            estimated_total: Some(0),
        }
    }

    /// Register a constant node (an externally-set source cell with no async
    /// feed). Mutate it via [`Session::cell_mut`] + [`Session::flush`]; wire it
    /// downstream via the deref'd plain handle (`*h`).
    pub fn add_const<T: Send + Sync + 'static>(&mut self, value: T) -> SourceHandle<RefSource<T>> {
        self.builder.push_source(RefSource::new(value))
    }

    /// Register a [`Segment`] (most operators implement
    /// [`flowgraph::typed::Operator`], which is one). `inputs` are the upstream
    /// handles, shaped like the segment's input tree.
    pub fn add_operator<O>(
        &mut self,
        operator: O,
        inputs: <O::Inputs as InterfaceHandles>::Handles<'_>,
    ) -> <O::Outputs as InterfaceHandles>::HandlesOwned
    where
        O: Segment,
        O::Inputs: InterfaceHandles,
        O::Outputs: InterfaceHandles,
    {
        self.builder.push(operator, inputs)
    }

    /// Register a [`Record`] for a data stream, wiring the driver's [`Clock`]
    /// so each recorded row is stamped with the current event time. `Record`
    /// needs the clock, so it cannot go through [`add_operator`](Self::add_operator).
    pub fn add_record<T: Scalar>(&mut self, data: Handle<RefPort<Array<T>>>) -> Handle<RefPort<Series<T>>> {
        self.builder.push(Record::new(self.clock.clone()), data)
    }

    /// The driver's [`Clock`]. For building custom time-reading operators (held
    /// in their own state, like [`Record`]); ordinary operators are clock-free.
    pub fn clock(&self) -> Clock {
        self.clock.clone()
    }

    /// Register a Python operator in **return mode**.
    /// `source` is a Python expression evaluating to a callable that takes
    /// `inputs.len()` `float64` ndarrays and returns a 1-D `float64` ndarray of
    /// length `out_len`, e.g. `"lambda a, b: a + b"`. Operators run on the shared
    /// (free-threaded) interpreter and execute in parallel on the pool. See the
    /// `operators::python` module docs for the data model, retention contract, and setup.
    pub fn add_py_operator(
        &mut self,
        source: &str,
        inputs: &[Handle<RefPort<Array<f64>>>],
        out_len: usize,
    ) -> Handle<RefPort<Array<f64>>> {
        self.builder
            .push(crate::operators::PyOperator::new(source, out_len), inputs)
    }

    /// Register a Python operator in **write mode** (zero-copy output). `source`
    /// evaluates to a callable `f(out, *inputs)` taking a writable `float64`
    /// ndarray `out` (aliasing the output buffer) followed by the input ndarrays;
    /// it writes its `out_len` results into `out` in place and its return value is
    /// ignored, e.g. `"lambda out, a, b: np.add(a, b, out=out)"`.
    /// The `out` view must not escape the call (see the `operators::python` module docs).
    pub fn add_py_operator_writing(
        &mut self,
        source: &str,
        inputs: &[Handle<RefPort<Array<f64>>>],
        out_len: usize,
    ) -> Handle<RefPort<Array<f64>>> {
        self.builder
            .push(crate::operators::PyOperator::writing(source, out_len), inputs)
    }

    /// Register a class-based Python operator. `source` is a Python program
    /// binding the operator instance to `__op__`, implementing the contract
    /// `init(inputs, timestamp) -> state` and
    /// `compute(state, inputs, output, timestamp, produced) -> bool`. The driver
    /// [`Clock`] is wired through so the operator sees event time. `out_shape` is
    /// the output element shape (`[]` for a scalar). See [`PyClassOperator`](crate::operators::PyClassOperator).
    pub fn add_py_class_operator(
        &mut self,
        source: &str,
        params: crate::operators::PyParams,
        inputs: &[Handle<RefPort<Array<f64>>>],
        out_shape: &[usize],
    ) -> Handle<RefPort<Array<f64>>> {
        self.builder.push(
            crate::operators::PyClassOperator::<flowgraph::typed::RefPorts<Array<f64>>>::from_source(
                source,
                params,
                out_shape.to_vec(),
                self.clock.clone(),
            ),
            inputs,
        )
    }

    /// Register a class-based Python operator loaded from a plain `.py` file.
    /// The file defines `build(**kwargs)` (called with `params`) or binds
    /// `__op__`; see [`PyClassOperator`](crate::operators::PyClassOperator).
    /// Convenience for all-`Array<f64>` inputs; heterogeneous (Array + Series)
    /// operators register via [`add_operator`](Self::add_operator) with
    /// `PyClassOperator::<I>::from_file(..)`.
    pub fn add_py_operator_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
        params: crate::operators::PyParams,
        inputs: &[Handle<RefPort<Array<f64>>>],
        out_shape: &[usize],
    ) -> Handle<RefPort<Array<f64>>> {
        self.builder.push(
            crate::operators::PyClassOperator::<flowgraph::typed::RefPorts<Array<f64>>>::from_file(
                path,
                params,
                out_shape.to_vec(),
                self.clock.clone(),
            ),
            inputs,
        )
    }

    /// Register a [`Source`]. Its output cell is a `RefSource` node; the async
    /// feed is wired up at [`build`](Self::build) time.
    pub fn add_source<S: Source>(
        &mut self,
        source: S,
        initial: S::Output,
    ) -> Handle<RefPort<S::Output>>
    where
        S::Output: Send + Sync + 'static,
    {
        let handle = self.builder.push_source(RefSource::new(initial));
        let erased = erased_flow_source(source);
        // Accumulate the progress estimate: sum the per-source counts; a single
        // un-estimable source makes the whole total unknown (None). Live/one-row-
        // per-event sources report their row count so `on_flush` tracks rows.
        self.estimated_total = match (self.estimated_total, erased.estimated_event_count) {
            (Some(acc), Some(n)) => Some(acc.saturating_add(n)),
            _ => None,
        };
        self.sources.push(SourceDescriptor {
            index: handle.index(),
            erased,
        });
        *handle
    }

    /// Finalize the graph and initialise every source (creating its channels),
    /// producing a runnable [`Session`] with a single-threaded pool.
    pub fn build(self) -> Session {
        self.build_with_threads(0)
    }

    /// Like [`build`](Self::build) but with `n` extra worker threads in the pool.
    pub fn build_with_threads(self, n: usize) -> Session {
        let Scenario {
            builder,
            clock,
            sources,
            estimated_total,
        } = self;
        let graph = Graph::from_builder(builder);

        let mut slots = Vec::with_capacity(sources.len());
        for desc in sources {
            let SourceDescriptor { index, erased } = desc;
            let (hist_rx_ptr, live_rx_ptr, state_ptr) = (erased.init_fn)(Instant::MIN);
            slots.push(SourceSlot {
                index,
                hist_rx_ptr,
                live_rx_ptr,
                state_ptr,
                poll_fn: erased.poll_fn,
                write_fn: erased.write_fn,
                rx_drop_fn: erased.rx_drop_fn,
                state_drop_fn: erased.state_drop_fn,
            });
        }

        Session {
            graph,
            pool: Pool::new(n),
            clock,
            slots,
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

/// A live execution: owns the graph, the worker pool, the clock, and the
/// per-source channel state.
pub struct Session {
    graph: Graph,
    pool: Pool,
    clock: Clock,
    slots: Vec<SourceSlot>,
    estimated_total: Option<usize>,
}

// SAFETY: the graph is Send+Sync; the pool and slots are Send; the clock is an
// Arc<AtomicI64>. The session is driven from a single task.
unsafe impl Send for Session {}

impl Session {
    /// Sum of every source's estimated event count, or `None` if any source
    /// couldn't estimate. With the panel sources emitting one event per row,
    /// this is the long-table row count; pair it with [`run`](Self::run)'s
    /// `on_flush` count (also rows) to drive a progress bar. Advisory.
    pub fn estimated_event_count(&self) -> Option<usize> {
        self.estimated_total
    }

    /// Immutable access to a by-reference node value.
    pub fn value<T: Send + Sync + 'static>(&self, handle: Handle<RefPort<T>>) -> &T {
        self.graph.ref_view(handle)
    }

    /// Mutable access to a source cell's value (marks its dirty cone). Only
    /// source cells ([`Scenario::add_const`] / [`Scenario::add_source`]) can be
    /// poked; operator outputs are state-owned and engine-written.
    pub fn cell_mut<T: Send + Sync + 'static>(
        &mut self,
        handle: SourceHandle<RefSource<T>>,
    ) -> &mut T {
        self.graph.state_mut(handle)
    }

    /// Manually advance the clock and stabilize (the synchronous-step API).
    /// Call after mutating one or more source cells via [`cell_mut`](Self::cell_mut).
    pub fn flush(&mut self, timestamp: Instant) {
        self.clock.set(timestamp);
        self.graph.stabilize(&mut self.pool);
    }

    /// Drive the unified event loop until every source is exhausted.
    ///
    /// `on_flush(batch_ts, events_processed_so_far)` is invoked once per
    /// coalesced batch.
    pub async fn run(&mut self, on_flush: impl FnMut(Instant, usize)) {
        self.run_with_shutdown(on_flush, Arc::new(AtomicBool::new(false)))
            .await;
    }

    /// Like [`run`](Self::run), but cooperatively cancellable.
    pub async fn run_with_shutdown(
        &mut self,
        mut on_flush: impl FnMut(Instant, usize),
        shutdown: ShutdownFlag,
    ) {
        let n = self.slots.len();
        if n == 0 {
            return;
        }

        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
        let mut hist_refills: FuturesUnordered<ErasedRecvFuture> = FuturesUnordered::new();
        let mut live_refills: FuturesUnordered<ErasedRecvFuture> = FuturesUnordered::new();
        let mut current_ts: Instant = Instant::MIN;
        let mut events_processed: usize = 0;

        let mut last_hist_ts: Vec<Instant> = vec![Instant::MIN; n];
        let mut hist_pending_ts: BTreeMap<Instant, usize> = BTreeMap::new();

        // Seed one hist + one live future per source.
        for i in 0..n {
            // SAFETY: the slots outlive every future created here.
            let slot = &self.slots[i];
            hist_refills.push(unsafe { slot.recv_future(i, ChannelKind::Hist) });
            *hist_pending_ts.entry(Instant::MIN).or_insert(0) += 1;
            live_refills.push(unsafe { slot.recv_future(i, ChannelKind::Live) });
        }

        drain_hist(&mut hist_refills, &mut heap, &mut last_hist_ts, &mut hist_pending_ts).await;
        drain_live(&mut live_refills, &mut heap, current_ts).await;

        let mut queue_ts: Option<Instant> = None;
        let mut pending = false;

        loop {
            drain_hist(&mut hist_refills, &mut heap, &mut last_hist_ts, &mut hist_pending_ts).await;
            drain_live(&mut live_refills, &mut heap, current_ts).await;

            let Some(&Reverse(HeapEntry { ts: min_ts, .. })) = heap.peek() else {
                if live_refills.is_empty() {
                    break;
                }
                while let Some((src, kind, opt_ts)) = live_refills.next().await {
                    if let Some(ts) = opt_ts {
                        let ts = ts.max(current_ts);
                        heap.push(Reverse(HeapEntry {
                            ts,
                            source_idx: src,
                            kind,
                        }));
                        break;
                    }
                    if live_refills.is_empty() {
                        break;
                    }
                }
                continue;
            };

            // Coalescing boundary: stabilize the previous batch when ts advances.
            // `pending` is only false before the very first batch; thereafter
            // the pop loop below always writes >=1 entry, so it stays true and
            // simply means "a batch is waiting to stabilize when ts advances".
            if pending && min_ts > queue_ts.unwrap() {
                let qts = queue_ts.unwrap();
                self.clock.set(qts);
                self.graph.stabilize(&mut self.pool);
                on_flush(qts, events_processed);
                if shutdown.load(Ordering::Relaxed) {
                    return;
                }
            }

            // Pop and apply every entry at min_ts.
            while let Some(&Reverse(HeapEntry { ts, .. })) = heap.peek() {
                if ts > min_ts {
                    break;
                }
                let Reverse(HeapEntry {
                    ts: _,
                    source_idx,
                    kind,
                }) = heap.pop().unwrap();

                // Write the buffered item into the source node's state via the
                // typed `state_mut` (marking the dirty cone). The item was
                // peeked, so `write_fn` finds it pending; it returns the number
                // of logical events it represented (a panel ships a whole
                // tick's rows as one item, so this is the row count, not 1).
                let slot = &self.slots[source_idx];
                let n = unsafe {
                    (slot.write_fn)(
                        slot.state_ptr,
                        slot.rx_ptr(kind),
                        &mut self.graph,
                        slot.index,
                        min_ts,
                    )
                };
                events_processed = events_processed.saturating_add(n);

                // Re-queue the consumed channel's future.
                let slot = &self.slots[source_idx];
                match kind {
                    ChannelKind::Hist => {
                        last_hist_ts[source_idx] = min_ts;
                        *hist_pending_ts.entry(min_ts).or_insert(0) += 1;
                        hist_refills.push(unsafe { slot.recv_future(source_idx, ChannelKind::Hist) });
                    }
                    ChannelKind::Live => {
                        live_refills.push(unsafe { slot.recv_future(source_idx, ChannelKind::Live) });
                    }
                }
            }
            current_ts = min_ts;
            queue_ts = Some(min_ts);
            pending = true;
        }

        // Final batch.
        if pending {
            let qts = queue_ts.unwrap();
            self.clock.set(qts);
            self.graph.stabilize(&mut self.pool);
            on_flush(qts, events_processed);
        }
    }
}

// ---------------------------------------------------------------------------
// drain_hist / drain_live (ported verbatim from scenario::queue)
// ---------------------------------------------------------------------------

async fn drain_hist(
    hist_refills: &mut FuturesUnordered<ErasedRecvFuture>,
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    last_hist_ts: &mut [Instant],
    hist_pending_ts: &mut BTreeMap<Instant, usize>,
) {
    while let Some((src, _kind, opt_ts)) = hist_refills.next().await {
        let lower = last_hist_ts[src];
        if let Some(cnt) = hist_pending_ts.get_mut(&lower) {
            *cnt -= 1;
            if *cnt == 0 {
                hist_pending_ts.remove(&lower);
            }
        }

        if let Some(ts) = opt_ts {
            last_hist_ts[src] = ts;
            heap.push(Reverse(HeapEntry {
                ts,
                source_idx: src,
                kind: ChannelKind::Hist,
            }));
        }

        if hist_refills.is_empty() {
            break;
        }

        let heap_min = heap.peek().map(|&Reverse(e)| e.ts).unwrap_or(Instant::MAX);
        let pending_lower = hist_pending_ts.keys().next().copied().unwrap_or(Instant::MAX);
        if pending_lower > heap_min {
            break;
        }
    }
}

async fn drain_live(
    live_refills: &mut FuturesUnordered<ErasedRecvFuture>,
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    current_ts: Instant,
) {
    poll_fn(|cx| {
        loop {
            match Pin::new(&mut *live_refills).poll_next_unpin(cx) {
                Poll::Ready(Some((src, kind, Some(ts)))) => {
                    let ts = ts.max(current_ts);
                    heap.push(Reverse(HeapEntry {
                        ts,
                        source_idx: src,
                        kind,
                    }));
                }
                Poll::Ready(Some((_src, _kind, None))) => {}
                Poll::Ready(None) | Poll::Pending => return Poll::Ready(()),
            }
        }
    })
    .await;
}

// ---------------------------------------------------------------------------
// Tests: reproduce the legacy `scenario_run_*` replay outcomes.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{Add, Filter};
    use crate::sources::ArraySource;
    use crate::{Array, Series};

    fn tss(xs: &[i64]) -> Vec<Instant> {
        xs.iter().copied().map(Instant::from_nanos).collect()
    }

    fn src(ts: &[i64], vals: &[f64]) -> ArraySource<f64> {
        ArraySource::new(
            Series::from_vec(&[], tss(ts), vals.to_vec()),
            Array::scalar(0.0),
        )
    }

    /// scenario_run_single_source: replay [10,20,30] @ [1,2,3] into a Record.
    #[tokio::test]
    async fn run_single_source_record() {
        let mut sc = Scenario::new();
        let h = sc.add_source(src(&[1, 2, 3], &[10.0, 20.0, 30.0]), Array::scalar(0.0));
        let hrec = sc.add_record(h);

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(s.values(), &[10.0, 20.0, 30.0]);
    }

    /// scenario_run_two_sources_add: staggered timestamps; the un-fired input
    /// keeps its stale value. ts1:10+0, ts2:10+20, ts3:30+40 → [10,30,70].
    #[tokio::test]
    async fn run_two_sources_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(src(&[1, 3], &[10.0, 30.0]), Array::scalar(0.0));
        let hb = sc.add_source(src(&[2, 3], &[20.0, 40.0]), Array::scalar(0.0));
        let ho = sc.add_operator(Add::<f64>::new(), (ha, hb));
        let hrec = sc.add_record(ho);

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(s.values(), &[10.0, 30.0, 70.0]);
    }

    /// scenario_run_coalescing: two sources at the same timestamps → one batch
    /// each. ts1:10+100, ts2:20+200 → [110,220].
    #[tokio::test]
    async fn run_coalescing() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(src(&[1, 2], &[10.0, 20.0]), Array::scalar(0.0));
        let hb = sc.add_source(src(&[1, 2], &[100.0, 200.0]), Array::scalar(0.0));
        let ho = sc.add_operator(Add::<f64>::new(), (ha, hb));
        let hrec = sc.add_record(ho);

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.timestamps(), tss(&[1, 2]).as_slice());
        assert_eq!(s.values(), &[110.0, 220.0]);
    }

    /// scenario_run_filter: the cutoff must survive the driver — dropped ticks
    /// produce no Record row. [1,5,2,10] keep >3 → (2,5),(4,10).
    #[tokio::test]
    async fn run_filter_cutoff() {
        let mut sc = Scenario::new();
        let h = sc.add_source(
            src(&[1, 2, 3, 4], &[1.0, 5.0, 2.0, 10.0]),
            Array::scalar(0.0),
        );
        let hf = sc.add_operator(Filter(|v: &Array<f64>| v[0] > 3.0), h);
        let hrec = sc.add_record(hf);

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.len(), 2);
        assert_eq!(s.timestamps(), tss(&[2, 4]).as_slice());
        assert_eq!(s.values(), &[5.0, 10.0]);
    }

    /// on_flush fires once per coalesced batch (3 distinct timestamps here).
    #[tokio::test]
    async fn on_flush_per_batch() {
        let mut sc = Scenario::new();
        let h = sc.add_source(src(&[1, 2, 3], &[10.0, 20.0, 30.0]), Array::scalar(0.0));
        let _ = sc.add_record(h);

        let mut session = sc.build();
        let mut batches = Vec::new();
        session.run(|ts, _| batches.push(ts.as_nanos())).await;
        assert_eq!(batches, vec![1, 2, 3]);
    }

    // -- Adversarial guards (ported from scenario::queue tests) --------------

    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    use flowgraph::typed::{Operator as FgOperator, RefPort};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tokio::sync::mpsc;

    /// A source backed by pre-filled bounded hist + live channels.
    #[derive(Clone)]
    struct PrefilledSource {
        hist_events: Vec<(Instant, f64)>,
        live_events: Vec<(Instant, f64)>,
    }

    impl Source for PrefilledSource {
        type Event = f64;
        type Output = Array<f64>;
        type State = ();

        fn init(
            &self,
            _ts: Instant,
        ) -> (
            mpsc::Receiver<(Instant, f64)>,
            mpsc::Receiver<(Instant, f64)>,
            Array<f64>,
            (),
        ) {
            let (hist_tx, hist_rx) = mpsc::channel(self.hist_events.len().max(1));
            for evt in &self.hist_events {
                hist_tx.try_send(*evt).unwrap();
            }
            drop(hist_tx);
            let (live_tx, live_rx) = mpsc::channel(self.live_events.len().max(1));
            for evt in &self.live_events {
                live_tx.try_send(*evt).unwrap();
            }
            drop(live_tx);
            (hist_rx, live_rx, Array::scalar(0.0), ())
        }

        fn write(_state: &mut (), event: f64, output: &mut Array<f64>, _ts: Instant) -> usize {
            output[0] = event;
            1
        }
    }

    /// Sink operator that records `(batch_ts, source_id)` in execution order —
    /// reads the [`Clock`] (held in its state, like `Record`) to check global
    /// ordering across sources.
    struct GlobalLogger {
        source_id: usize,
        log: Arc<Mutex<Vec<(i64, usize)>>>,
        clock: Clock,
    }

    impl FgOperator for GlobalLogger {
        type Inputs = RefPort<Array<f64>>;
        type Outputs = ();
        type State = (usize, Arc<Mutex<Vec<(i64, usize)>>>, Clock);

        fn init(self) -> Self::State {
            (self.source_id, self.log, self.clock)
        }

        fn compute<'a, 'b: 'a>(
            _: (bool, &'a Array<f64>),
            state: &'b mut Self::State,
            init: bool,
        ) {
            if !init {
                state.1.lock().unwrap().push((state.2.get().as_nanos(), state.0));
            }
        }

        fn passthrough<'a, 'b: 'a>(_: (bool, &'a Array<f64>), _: &'b Self::State) {}
    }

    type EventVec = Vec<(Instant, f64)>;

    fn generate_events(rng: &mut StdRng, source_id: usize) -> (EventVec, EventVec) {
        let hist_count: usize = rng.gen_range(0..=15);
        let live_count: usize = rng.gen_range(0..=15);

        let mut hist_ts: Vec<i64> = (0..hist_count).map(|_| rng.gen_range(0..=100)).collect();
        hist_ts.sort();
        let hist_max = hist_ts.last().copied().unwrap_or(0);
        let mut live_ts: Vec<i64> = (0..live_count)
            .map(|_| rng.gen_range(hist_max..=200))
            .collect();
        live_ts.sort();

        let hist: EventVec = hist_ts
            .into_iter()
            .enumerate()
            .map(|(i, t)| (Instant::from_nanos(t), source_id as f64 * 10000.0 + i as f64))
            .collect();
        let live: EventVec = live_ts
            .into_iter()
            .enumerate()
            .map(|(i, t)| (Instant::from_nanos(t), source_id as f64 * 10000.0 + 1000.0 + i as f64))
            .collect();
        (hist, live)
    }

    fn check_invariants(
        series: &Series<f64>,
        hist_events: &[(Instant, f64)],
        live_events: &[(Instant, f64)],
        source_id: usize,
        seed: u64,
    ) {
        let tsr = series.timestamps();
        for w in tsr.windows(2) {
            assert!(w[0] <= w[1], "seed {seed}, src {source_id}: monotonicity");
        }
        let distinct_hist: BTreeSet<Instant> = hist_events.iter().map(|e| e.0).collect();
        let recorded: BTreeSet<Instant> = tsr.iter().copied().collect();
        for &ht in &distinct_hist {
            assert!(recorded.contains(&ht), "seed {seed}, src {source_id}: hist {ht:?} missing");
        }
        let hist_expected: Vec<Instant> = distinct_hist.iter().copied().collect();
        if tsr.len() >= hist_expected.len() {
            assert_eq!(&tsr[..hist_expected.len()], &hist_expected[..], "seed {seed}, src {source_id}: hist prefix");
        }
        if !live_events.is_empty() {
            let min_live_ts = live_events[0].0;
            for &t in tsr.iter().skip(hist_expected.len()) {
                assert!(t >= min_live_ts, "seed {seed}, src {source_id}: live before min");
            }
        }
        assert!(tsr.len() >= distinct_hist.len(), "seed {seed}, src {source_id}: too few");
        if !tsr.is_empty() {
            let min_original = hist_events
                .iter()
                .chain(live_events.iter())
                .map(|e| e.0)
                .min()
                .unwrap();
            assert!(tsr[0] >= min_original, "seed {seed}, src {source_id}: fabricated");
        }
    }

    /// 100 random scenarios interleaving hist + live across multiple sources.
    #[tokio::test]
    async fn random_interleaving() {
        for seed in 0..100u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let n_sources: usize = rng.gen_range(1..=4);

            let mut sc = Scenario::new();
            let log = Arc::new(Mutex::new(Vec::new()));
            let mut source_data: Vec<(EventVec, EventVec)> = Vec::new();
            let mut records = Vec::new();

            for i in 0..n_sources {
                let (hist, live) = generate_events(&mut rng, i);
                let h = sc.add_source(
                    PrefilledSource {
                        hist_events: hist.clone(),
                        live_events: live.clone(),
                    },
                    Array::scalar(0.0),
                );
                records.push(sc.add_record(h));
                let clk = sc.clock();
                sc.add_operator(
                    GlobalLogger {
                        source_id: i,
                        log: log.clone(),
                        clock: clk,
                    },
                    h,
                );
                source_data.push((hist, live));
            }

            let mut session = sc.build();
            session.run(|_, _| {}).await;

            for (i, (hr, (hist, live))) in records.iter().zip(source_data.iter()).enumerate() {
                let series: &Series<f64> = session.value(*hr);
                check_invariants(series, hist, live, i, seed);
            }
            // Global monotonicity: the logged batch timestamps are non-decreasing.
            let log = log.lock().unwrap();
            for w in log.windows(2) {
                assert!(w[0].0 <= w[1].0, "seed {seed}: global monotonicity");
            }
        }
    }

    /// A stale live event (ts=50) arriving after hist advances current_ts to
    /// 100 must be clamped to 100 and coalesced, not ingested at 50.
    #[tokio::test]
    async fn live_clamping() {
        type ChannelPair = (mpsc::Receiver<(Instant, f64)>, mpsc::Receiver<(Instant, f64)>);

        #[derive(Clone)]
        struct ManualChannel {
            channels: Arc<Mutex<Option<ChannelPair>>>,
        }

        impl Source for ManualChannel {
            type Event = f64;
            type Output = Array<f64>;
            type State = ();

            fn init(
                &self,
                _ts: Instant,
            ) -> (
                mpsc::Receiver<(Instant, f64)>,
                mpsc::Receiver<(Instant, f64)>,
                Array<f64>,
                (),
            ) {
                let (hist_rx, live_rx) = self.channels.lock().unwrap().take().expect("init once");
                (hist_rx, live_rx, Array::scalar(0.0), ())
            }

            fn write(_state: &mut (), event: f64, output: &mut Array<f64>, _ts: Instant) -> usize {
                output[0] = event;
                1
            }
        }

        let (hist_tx, hist_rx) = mpsc::channel(1);
        let (live_tx, live_rx) = mpsc::channel(1);
        hist_tx.send((Instant::from_nanos(100), 1.0)).await.unwrap();
        drop(hist_tx);

        let mut sc = Scenario::new();
        let hs = sc.add_source(
            ManualChannel {
                channels: Arc::new(Mutex::new(Some((hist_rx, live_rx)))),
            },
            Array::scalar(0.0),
        );
        let hrec = sc.add_record(hs);

        tokio::spawn(async move {
            live_tx.send((Instant::from_nanos(50), 2.0)).await.unwrap();
        });

        let mut session = sc.build();
        session.run(|_, _| {}).await;

        let s: &Series<f64> = session.value(hrec);
        assert_eq!(s.len(), 1);
        assert_eq!(s.timestamps(), &[Instant::from_nanos(100)]);
    }
}
