//! Scenario — the DAG runtime that owns nodes and dispatches operators.
//!
//! # Architecture
//!
//! Every node has an [`Observable`] (always present) and an optional
//! [`Series`] (allocated on demand via [`materialize`]).  Operators write into
//! the observable; the scenario copies to the series if materialised.
//!
//! All nodes are stored as type-erased [`NodeSlot`]s in a flat `Vec`.
//! Each operator is wrapped in an [`OperatorSlot`] with pre-cast I/O raw
//! pointers and a monomorphised `compute_fn` function pointer.
//!
//! Type safety is enforced at registration time via [`ObservableHandle<T>`]
//! and [`SeriesHandle<T>`] generics.  After registration the scenario
//! operates on raw pointers only.
//!
//! # Flush algorithm
//!
//! On each tick the caller writes to source observables and calls [`flush`].
//! A min-heap processes only the operators reachable from the updated sources,
//! in registration order (which is topological order, since inputs must exist
//! before an operator can be registered).  This is O(active) not O(total).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use std::future::Future;
use std::pin::Pin;

use crate::observable::Observable;
use crate::operator::Operator;
use crate::refs::{InputRef, InputRefs, OutputRef};
use crate::series::Series;
use crate::source::Source;

/// A typed handle into a [`Scenario`]'s node storage.
#[derive(Debug, Clone, Copy)]
pub struct ObservableHandle<T: Copy> {
    index: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> ObservableHandle<T> {
    pub(self) fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy> ObservableHandle<T> {
    pub fn index(&self) -> usize {
        self.index
    }
}

/// A typed handle into a [`Scenario`]'s node storage.
#[derive(Debug, Clone, Copy)]
pub struct SeriesHandle<T: Copy> {
    index: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> SeriesHandle<T> {
    pub(self) fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy> SeriesHandle<T> {
    pub fn index(&self) -> usize {
        self.index
    }
}

// ---------------------------------------------------------------------------
// InputHandle / InputHandles — maps InputRef types to Scenario handles
// ---------------------------------------------------------------------------

/// Maps an [`InputRef`] to its corresponding Scenario handle type.
///
/// # Safety
///
/// `node_id` must return the correct `(node_index, is_series)` pair for the
/// handle.
pub unsafe trait InputHandle<'a>: InputRef<'a> {
    /// The handle type used at registration time.
    type Handle;

    /// Extract `(node_index, is_series)` from a handle.
    fn node_id(handle: &Self::Handle) -> (usize, bool);
}

unsafe impl<'a, T: Copy + 'static> InputHandle<'a> for &'a Observable<T> {
    type Handle = ObservableHandle<T>;

    #[inline(always)]
    fn node_id(handle: &ObservableHandle<T>) -> (usize, bool) {
        (handle.index(), false)
    }
}

unsafe impl<'a, T: Copy + 'static> InputHandle<'a> for &'a Series<T> {
    type Handle = SeriesHandle<T>;

    #[inline(always)]
    fn node_id(handle: &SeriesHandle<T>) -> (usize, bool) {
        (handle.index(), true)
    }
}

/// Maps an [`InputRefs`] collection to its corresponding Handles collection.
///
/// # Safety
///
/// `node_ids` must return correct `(node_index, is_series)` pairs.
pub unsafe trait InputHandles<'a>: InputRefs<'a> {
    /// The handle collection type used at registration time.
    type Handles;

    /// Extract `(node_index, is_series)` from each handle.
    fn node_ids(handles: &Self::Handles) -> Box<[(usize, bool)]>;
}

unsafe impl<'a, R: InputHandle<'a>> InputHandles<'a> for Box<[R]> {
    type Handles = Box<[R::Handle]>;

    fn node_ids(handles: &Box<[R::Handle]>) -> Box<[(usize, bool)]> {
        handles.iter().map(|h| R::node_id(h)).collect()
    }
}

macro_rules! impl_tuple {
    ($($idx:tt: $R:ident),+ $(,)?) => {
        unsafe impl<'a, $($R: InputHandle<'a>),+> InputHandles<'a> for ($($R,)+) {
            type Handles = ($($R::Handle,)+);

            fn node_ids(handles: &Self::Handles) -> Box<[(usize, bool)]> {
                Box::new([$($R::node_id(&handles.$idx)),+])
            }
        }
    };
}

impl_tuple!(0: A);
impl_tuple!(0: A, 1: B);
impl_tuple!(0: A, 1: B, 2: C);
impl_tuple!(0: A, 1: B, 2: C, 3: D);
impl_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

// ---------------------------------------------------------------------------
// NodeSlot (type-erased)
// ---------------------------------------------------------------------------

/// Type-erased node: observable + optional series.
pub struct NodeSlot {
    pub(crate) obs_ptr: *mut u8,
    pub(crate) series_ptr: *mut u8,
    /// Copy observable value into series (typed).
    materialize_fn: unsafe fn(*mut u8, *mut u8, i64),
    drop_fn: unsafe fn(*mut u8, *mut u8),
    /// Element shape of the observable.
    shape: Box<[usize]>,
    /// Stride of the observable (product of shape dims, min 1).
    stride: usize,
}

impl Drop for NodeSlot {
    fn drop(&mut self) {
        unsafe { (self.drop_fn)(self.obs_ptr, self.series_ptr) }
    }
}

/// Type-erased materialise: copy observable value into series.
unsafe fn materialize_copy<T: Copy>(obs_ptr: *mut u8, series_ptr: *mut u8, timestamp: i64) {
    let obs = unsafe { &*(obs_ptr as *const Observable<T>) };
    let series = unsafe { &mut *(series_ptr as *mut Series<T>) };
    series.push(timestamp, obs.current());
}

/// Drop both the observable and the optional series.
unsafe fn drop_node<T: Copy>(obs_ptr: *mut u8, series_ptr: *mut u8) {
    unsafe { drop(Box::from_raw(obs_ptr as *mut Observable<T>)) };
    if !series_ptr.is_null() {
        unsafe { drop(Box::from_raw(series_ptr as *mut Series<T>)) };
    }
}

// ---------------------------------------------------------------------------
// OperatorSlot (type-erased)
// ---------------------------------------------------------------------------

/// Type-erased operator with pre-computed I/O pointers.
pub(crate) struct OperatorSlot {
    pub(crate) output_node_index: usize,
    input_ptrs: Vec<*const u8>,
    output_obs_ptr: *mut u8,
    output_series_ptr: *mut u8,
    materialize_fn: unsafe fn(*mut u8, *mut u8, i64),
    compute_fn: unsafe fn(&[*const u8], *mut u8, *mut u8) -> bool,
    state: *mut u8,
    drop_fn: unsafe fn(*mut u8),
}

impl Drop for OperatorSlot {
    fn drop(&mut self) {
        unsafe { (self.drop_fn)(self.state) }
    }
}

// ---------------------------------------------------------------------------
// Type-erased compute for slice-input operators (&[S::Ref<'_>])
// ---------------------------------------------------------------------------

/// Type-erased compute entry point, monomorphised for each `Op`.
///
/// Uses [`InputRefs::from_raw`] and [`OutputRef::from_raw`] to reconstruct
/// the typed references.
#[inline]
unsafe fn erased_compute<Op>(
    input_ptrs: &[*const u8],
    output_ptr: *mut u8,
    state_ptr: *mut u8,
) -> bool
where
    Op: Operator,
{
    let state = unsafe { &mut *(state_ptr as *mut Op::State) };
    let inputs = unsafe { Op::Inputs::from_raw(input_ptrs) };
    let output = unsafe { Op::Output::from_raw(output_ptr) };
    Op::compute(state, inputs, output)
}

/// Drop function for heterogeneous operator slots.
unsafe fn drop_erased_op<Op: Operator>(state: *mut u8) {
    unsafe { drop(Box::from_raw(state as *mut Op::State)) };
}

// ---------------------------------------------------------------------------
// SourceSlot (type-erased) — stored at registration, consumed during run()
// ---------------------------------------------------------------------------

/// Stored at registration time.  The `setup_fn` closure captures the typed
/// source and is called during [`run()`] to subscribe and build a
/// [`SourceRuntime`].
struct SourceSlot {
    node_index: usize,
    /// Called during run(): subscribes the source and returns a SourceRuntime.
    setup_fn: Box<dyn FnOnce(*mut u8) -> Pin<Box<dyn Future<Output = SourceRuntime>>>>,
}

/// Created during run() after subscribing.  Holds type-erased channel state
/// and monomorphized function pointers for wait/write.
struct SourceRuntime {
    node_index: usize,
    obs_ptr: *mut u8,
    state: *mut u8,
    wait_historical_fn: unsafe fn(*mut u8) -> Pin<Box<dyn Future<Output = Option<i64>>>>,
    wait_live_fn: unsafe fn(*mut u8) -> Pin<Box<dyn Future<Output = Option<i64>>>>,
    write_historical_fn: unsafe fn(*mut u8, *mut u8) -> bool,
    write_live_fn: unsafe fn(*mut u8, *mut u8) -> bool,
    drop_fn: unsafe fn(*mut u8),
    pending_hist_ts: Option<i64>,
    hist_exhausted: bool,
    live_exhausted: bool,
}

impl Drop for SourceRuntime {
    fn drop(&mut self) {
        unsafe { (self.drop_fn)(self.state) }
    }
}

/// Per-source channel state, generic over the payload type `P`.
struct SourceChannelState<P> {
    hist_rx: tokio::sync::mpsc::Receiver<(i64, P)>,
    live_rx: tokio::sync::mpsc::Receiver<(i64, P)>,
    pending_hist: Option<(i64, P)>,
    pending_live: Option<(i64, P)>,
}

unsafe fn erased_wait_historical<S: Source>(
    state: *mut u8,
) -> Pin<Box<dyn Future<Output = Option<i64>>>> {
    Box::pin(async move {
        let s = unsafe { &mut *(state as *mut SourceChannelState<S::Event>) };
        if let Some(ref item) = s.pending_hist {
            return Some(item.0);
        }
        match s.hist_rx.recv().await {
            Some(item) => {
                let ts = item.0;
                s.pending_hist = Some(item);
                Some(ts)
            }
            None => None,
        }
    })
}

unsafe fn erased_wait_live<S: Source>(
    state: *mut u8,
) -> Pin<Box<dyn Future<Output = Option<i64>>>> {
    Box::pin(async move {
        let s = unsafe { &mut *(state as *mut SourceChannelState<S::Event>) };
        if let Some(ref item) = s.pending_live {
            return Some(item.0);
        }
        match s.live_rx.recv().await {
            Some(item) => {
                let ts = item.0;
                s.pending_live = Some(item);
                Some(ts)
            }
            None => None,
        }
    })
}

unsafe fn erased_write_historical<S: Source>(state: *mut u8, obs_ptr: *mut u8) -> bool {
    let s = unsafe { &mut *(state as *mut SourceChannelState<S::Event>) };
    if let Some((_, payload)) = s.pending_hist.take() {
        let output = unsafe { <S::Output<'_>>::from_raw(obs_ptr) };
        S::write(payload, output)
    } else {
        false
    }
}

unsafe fn erased_write_live<S: Source>(state: *mut u8, obs_ptr: *mut u8) -> bool {
    let s = unsafe { &mut *(state as *mut SourceChannelState<S::Event>) };
    if let Some((_, payload)) = s.pending_live.take() {
        let output = unsafe { <S::Output<'_>>::from_raw(obs_ptr) };
        S::write(payload, output)
    } else {
        false
    }
}

unsafe fn drop_source_channel_state<P>(state: *mut u8) {
    unsafe { drop(Box::from_raw(state as *mut SourceChannelState<P>)) };
}

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

/// Owns all nodes, operators, and sources; coordinates DAG execution.
///
/// Operators are registered in dependency order (inputs must already exist),
/// so operator indices are always in topological order.
pub struct Scenario {
    nodes: Vec<NodeSlot>,
    /// `edges[node_idx]` → operator indices that read from this node.
    edges: Vec<Vec<usize>>,
    operators: Vec<OperatorSlot>,
    // Reusable per-flush scratch space:
    pending: Vec<bool>,
    heap: BinaryHeap<Reverse<usize>>,
    // Sources for run():
    source_slots: Vec<SourceSlot>,
}

impl Scenario {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            operators: Vec::new(),
            pending: Vec::new(),
            heap: BinaryHeap::new(),
            source_slots: Vec::new(),
        }
    }

    // -- Private: node creation ---------------------------------------------

    /// Internal helper — create a node with an explicit initial value.
    fn create_node<T: Copy>(&mut self, shape: &[usize], initial: &[T]) -> ObservableHandle<T> {
        let idx = self.nodes.len();
        let obs = Box::new(Observable::<T>::new(shape, initial));
        let stride = obs.stride();
        self.nodes.push(NodeSlot {
            obs_ptr: Box::into_raw(obs) as *mut u8,
            series_ptr: std::ptr::null_mut(),
            materialize_fn: materialize_copy::<T>,
            drop_fn: drop_node::<T>,
            shape: shape.into(),
            stride,
        });
        self.edges.push(Vec::new());
        ObservableHandle::new(idx)
    }

    // -- Source registration ------------------------------------------------

    /// Register a source node with an explicit initial value.
    ///
    /// The initial value is written to the observable immediately.  Operator
    /// Operator outputs are initialised at registration time by
    /// [`add_operator`].
    pub fn add_source<T: Copy>(&mut self, shape: &[usize], initial: &[T]) -> ObservableHandle<T> {
        let idx = self.nodes.len();
        let obs = Box::new(Observable::<T>::new(shape, initial));
        let stride = obs.stride();
        self.nodes.push(NodeSlot {
            obs_ptr: Box::into_raw(obs) as *mut u8,
            series_ptr: std::ptr::null_mut(),
            materialize_fn: materialize_copy::<T>,
            drop_fn: drop_node::<T>,
            shape: shape.into(),
            stride,
        });
        self.edges.push(Vec::new());
        ObservableHandle::new(idx)
    }

    // -- Materialization ----------------------------------------------------

    /// Materialise a node: allocate a [`Series`] alongside the observable.
    ///
    /// Returns a [`SeriesHandle`] proving that the node has history storage.
    /// Panics if the node is already materialised.
    pub fn materialize<T: Copy>(&mut self, h: ObservableHandle<T>) -> SeriesHandle<T> {
        assert!(
            self.nodes[h.index].series_ptr.is_null(),
            "node already materialised"
        );
        let obs = unsafe { &*(self.nodes[h.index].obs_ptr as *const Observable<T>) };
        let shape = obs.shape().to_vec();
        let series = Box::new(Series::<T>::new(&shape, obs.current()));
        self.nodes[h.index].series_ptr = Box::into_raw(series) as *mut u8;
        // Update any existing operator that outputs to this node.
        for slot in &mut self.operators {
            if slot.output_node_index == h.index {
                slot.output_series_ptr = self.nodes[h.index].series_ptr;
            }
        }
        SeriesHandle::new(h.index)
    }

    // -- Observable / Series access -----------------------------------------

    /// Get a mutable reference to the concrete `Observable<T>` behind a
    /// handle.
    ///
    /// # Safety
    ///
    /// The handle must have been created by this scenario with the same `T`.
    #[inline(always)]
    pub unsafe fn observable_mut<T: Copy>(&mut self, h: ObservableHandle<T>) -> &mut Observable<T> {
        unsafe { &mut *(self.nodes[h.index].obs_ptr as *mut Observable<T>) }
    }

    /// Get a shared reference to the concrete `Observable<T>`.
    #[inline(always)]
    pub unsafe fn observable_ref<T: Copy>(&self, h: ObservableHandle<T>) -> &Observable<T> {
        unsafe { &*(self.nodes[h.index].obs_ptr as *const Observable<T>) }
    }

    /// Get a shared reference to the concrete `Series<T>` behind a handle.
    ///
    /// # Safety
    ///
    /// The handle must have been created by this scenario with the same `T`.
    #[inline(always)]
    pub unsafe fn series_ref<T: Copy>(&self, h: SeriesHandle<T>) -> &Series<T> {
        unsafe { &*(self.nodes[h.index].series_ptr as *const Series<T>) }
    }

    /// Get a mutable reference to the concrete `Series<T>`.
    #[inline(always)]
    pub unsafe fn series_mut<T: Copy>(&mut self, h: SeriesHandle<T>) -> &mut Series<T> {
        unsafe { &mut *(self.nodes[h.index].series_ptr as *mut Series<T>) }
    }

    // -- Operator registration -----------------------------------------------

    /// Register an operator.
    ///
    /// Creates the output node, runs an initial `compute` from the current
    /// input values, and returns a handle to the output observable.
    pub fn add_operator<'a, Op, T: Copy>(
        &'a mut self,
        inputs: impl Into<<Op::Inputs<'a> as InputHandles<'a>>::Handles>,
        op: Op,
    ) -> ObservableHandle<T>
    where
        Op: Operator,
        for<'b> Op::Inputs<'b>: InputHandles<'b>,
        for<'b> Op::Output<'b>: OutputRef<'b, Scalar = T>,
    {
        let handles = inputs.into();
        let node_ids = <Op::Inputs<'_> as InputHandles<'_>>::node_ids(&handles);
        let input_shapes: Box<[&[usize]]> = node_ids
            .iter()
            .map(|&(i, _)| &*self.nodes[i].shape)
            .collect();
        let output_shape = op.shape(&input_shapes);
        let state = op.init();

        // Create output node with zeroed buffer (overwritten by initial compute).
        let stride = output_shape.iter().product::<usize>();
        let zeroed = vec![unsafe { std::mem::zeroed::<T>() }; stride];
        let output = self.create_node::<T>(&output_shape, &zeroed);
        let op_idx = self.operators.len();

        let mut ptrs: Vec<*const u8> = Vec::with_capacity(node_ids.len());
        for &(i, is_series) in node_ids.iter() {
            self.edges[i].push(op_idx);
            if is_series {
                ptrs.push(self.nodes[i].series_ptr);
            } else {
                ptrs.push(self.nodes[i].obs_ptr);
            }
        }

        self.operators.push(OperatorSlot {
            output_node_index: output.index,
            input_ptrs: ptrs,
            output_obs_ptr: self.nodes[output.index].obs_ptr,
            output_series_ptr: self.nodes[output.index].series_ptr,
            materialize_fn: self.nodes[output.index].materialize_fn,
            compute_fn: erased_compute::<Op>,
            state: Box::into_raw(Box::new(state)) as *mut u8,
            drop_fn: drop_erased_op::<Op>,
        });

        self.pending.push(false);

        // Compute initial output value from current input values.
        let slot = &self.operators[op_idx];
        unsafe {
            (slot.compute_fn)(&slot.input_ptrs, slot.output_obs_ptr, slot.state);
        }

        output
    }

    // -- Bridge helpers (for PyO3 bridge) ------------------------------------

    /// Register an operator with a custom compute function and type-erased
    /// state.  The output node type is determined by the caller via
    /// `create_node_fn` which should call `self.create_node::<T>()` for the
    /// appropriate `T`.
    ///
    /// Returns the output node index.
    pub(crate) fn add_raw_operator(
        &mut self,
        input_indices: &[usize],
        output_node_index: usize,
        compute_fn: unsafe fn(&[*const u8], *mut u8, *mut u8) -> bool,
        state: *mut u8,
        drop_fn: unsafe fn(*mut u8),
    ) {
        let op_idx = self.operators.len();

        let mut ptrs: Vec<*const u8> = Vec::with_capacity(input_indices.len());
        for &idx in input_indices {
            self.edges[idx].push(op_idx);
            ptrs.push(self.nodes[idx].obs_ptr);
        }

        self.operators.push(OperatorSlot {
            output_node_index,
            input_ptrs: ptrs,
            output_obs_ptr: self.nodes[output_node_index].obs_ptr,
            output_series_ptr: self.nodes[output_node_index].series_ptr,
            materialize_fn: self.nodes[output_node_index].materialize_fn,
            compute_fn,
            state,
            drop_fn,
        });
        self.pending.push(false);
    }

    /// Create a typed node (for bridge dtype dispatch).
    pub(crate) fn create_node_typed<T: Copy>(&mut self, shape: &[usize], initial: &[T]) -> usize {
        self.create_node::<T>(shape, initial).index
    }

    /// Create a typed source node with an initial value (for bridge dtype
    /// dispatch).
    pub(crate) fn add_source_typed<T: Copy>(&mut self, shape: &[usize], initial: &[T]) -> usize {
        self.add_source::<T>(shape, initial).index
    }

    /// Register a source by node index (for bridge dtype dispatch).
    pub(crate) fn register_source_by_index<S, T: Copy + 'static>(
        &mut self,
        node_index: usize,
        source: S,
    ) where
        S: Source,
        for<'a> S::Output<'a>: OutputRef<'a, Scalar = T>,
    {
        let setup_fn: Box<dyn FnOnce(*mut u8) -> Pin<Box<dyn Future<Output = SourceRuntime>>>> =
            Box::new(move |obs_ptr| {
                Box::pin(async move {
                    let (hist_rx, live_rx) = Box::new(source).subscribe().await;
                    let state = Box::new(SourceChannelState {
                        hist_rx,
                        live_rx,
                        pending_hist: None,
                        pending_live: None,
                    });
                    SourceRuntime {
                        node_index,
                        obs_ptr,
                        state: Box::into_raw(state) as *mut u8,
                        wait_historical_fn: erased_wait_historical::<S>,
                        wait_live_fn: erased_wait_live::<S>,
                        write_historical_fn: erased_write_historical::<S>,
                        write_live_fn: erased_write_live::<S>,
                        drop_fn: drop_source_channel_state::<S::Event>,
                        pending_hist_ts: None,
                        hist_exhausted: false,
                        live_exhausted: false,
                    }
                })
            });
        self.source_slots.push(SourceSlot {
            node_index,
            setup_fn,
        });
    }

    /// Materialise a node by index (for bridge — no typed handle).
    pub(crate) fn materialize_by_index<T: Copy>(&mut self, node_index: usize) {
        assert!(
            self.nodes[node_index].series_ptr.is_null(),
            "node already materialised"
        );
        let obs = unsafe { &*(self.nodes[node_index].obs_ptr as *const Observable<T>) };
        let shape = obs.shape().to_vec();
        let series = Box::new(Series::<T>::new(&shape, obs.current()));
        self.nodes[node_index].series_ptr = Box::into_raw(series) as *mut u8;
        for slot in &mut self.operators {
            if slot.output_node_index == node_index {
                slot.output_series_ptr = self.nodes[node_index].series_ptr;
            }
        }
    }

    /// Number of nodes in the graph.
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Raw observable pointer for a node.
    pub(crate) fn node_obs_ptr(&self, idx: usize) -> *mut u8 {
        self.nodes[idx].obs_ptr
    }

    /// Raw series pointer for a node (null if not materialised).
    pub(crate) fn node_series_ptr(&self, idx: usize) -> *mut u8 {
        self.nodes[idx].series_ptr
    }

    /// Stride (number of elements per value) for a node.
    pub(crate) fn node_stride(&self, idx: usize) -> usize {
        self.nodes[idx].stride
    }

    // -- Source registration ------------------------------------------------

    /// Associate a [`Source`] with an existing source node.
    ///
    /// The source will be consumed by [`run`] to feed data into the node.
    pub fn register_source<S, T: Copy + 'static>(&mut self, handle: ObservableHandle<T>, source: S)
    where
        S: Source,
        for<'a> S::Output<'a>: OutputRef<'a, Scalar = T>,
    {
        self.register_source_by_index::<S, T>(handle.index, source);
    }

    // -- Async run (POCQ) ---------------------------------------------------

    /// Consume all registered sources, coalesce events, and propagate through
    /// the operator DAG.
    ///
    /// Implements the Point-of-Coherency Queue (POCQ) algorithm:
    /// - Historical constraint: all active historical sources must have a
    ///   pending event before advancing.
    /// - Coalescing: events at the same timestamp accumulate; flush triggers
    ///   when a strictly larger timestamp arrives.
    pub async fn run(&mut self) {
        // Subscribe all sources and build per-source runtime state.
        let slots: Vec<SourceSlot> = self.source_slots.drain(..).collect();
        let mut runtimes: Vec<SourceRuntime> = Vec::with_capacity(slots.len());
        for slot in slots {
            let obs_ptr = self.nodes[slot.node_index].obs_ptr;
            runtimes.push((slot.setup_fn)(obs_ptr).await);
        }

        // Fetch initial historical event from each source.
        for rt in &mut runtimes {
            if !rt.hist_exhausted {
                let ts = unsafe { (rt.wait_historical_fn)(rt.state) }.await;
                rt.pending_hist_ts = ts;
                if ts.is_none() {
                    rt.hist_exhausted = true;
                }
            }
        }

        // POCQ state.
        let mut queue_ts: Option<i64> = None;
        let mut queue_sources: Vec<usize> = Vec::new();

        loop {
            // Historical constraint: wait for all non-exhausted historical sources.
            for rt in &mut runtimes {
                if !rt.hist_exhausted && rt.pending_hist_ts.is_none() {
                    let ts = unsafe { (rt.wait_historical_fn)(rt.state) }.await;
                    rt.pending_hist_ts = ts;
                    if ts.is_none() {
                        rt.hist_exhausted = true;
                    }
                }
            }

            // Find the minimum pending timestamp across all sources.
            let min_ts = runtimes.iter().filter_map(|rt| rt.pending_hist_ts).min();

            let Some(min_ts) = min_ts else {
                // All historical sources exhausted — flush remaining queue and exit.
                // TODO: live source handling would go here.
                if !queue_sources.is_empty() {
                    self.flush(queue_ts.unwrap(), &queue_sources);
                    queue_sources.clear();
                }
                break;
            };

            // If the new timestamp is strictly greater than the current queue,
            // flush the accumulated events first.
            if let Some(qts) = queue_ts {
                if min_ts > qts {
                    self.flush(qts, &queue_sources);
                    queue_sources.clear();
                }
            }

            // Collect all sources at min_ts: write to observables, add to queue.
            for rt in &mut runtimes {
                if rt.pending_hist_ts == Some(min_ts) {
                    // Write the pending payload to the observable.
                    unsafe { (rt.write_historical_fn)(rt.state, rt.obs_ptr) };
                    queue_sources.push(rt.node_index);
                    rt.pending_hist_ts = None;

                    // Fetch next historical event from this source.
                    if !rt.hist_exhausted {
                        let ts = unsafe { (rt.wait_historical_fn)(rt.state) }.await;
                        rt.pending_hist_ts = ts;
                        if ts.is_none() {
                            rt.hist_exhausted = true;
                        }
                    }
                }
            }
            queue_ts = Some(min_ts);
        }
    }

    // -- Execution ----------------------------------------------------------

    /// Propagate updates through the DAG.
    ///
    /// After writing to source observables, call this with the indices of the
    /// updated source nodes.  Only operators reachable from those nodes will
    /// execute, in topological order.
    #[inline]
    pub fn flush(&mut self, timestamp: i64, updated_sources: &[usize]) {
        // Materialise updated source nodes (copy obs → series if materialised).
        for &idx in updated_sources {
            let node = &self.nodes[idx];
            if !node.series_ptr.is_null() {
                unsafe {
                    (node.materialize_fn)(node.obs_ptr, node.series_ptr, timestamp);
                }
            }
        }
        // Seed the min-heap with operators directly downstream of updated sources.
        for &idx in updated_sources {
            for &op_idx in &self.edges[idx] {
                if !self.pending[op_idx] {
                    self.pending[op_idx] = true;
                    self.heap.push(Reverse(op_idx));
                }
            }
        }
        // Process in topological order (operator index IS topological rank).
        while let Some(Reverse(op_idx)) = self.heap.pop() {
            self.pending[op_idx] = false;
            let slot = &self.operators[op_idx];
            // SAFETY: pointers were set up correctly at registration time.
            let produced =
                unsafe { (slot.compute_fn)(&slot.input_ptrs, slot.output_obs_ptr, slot.state) };
            if produced {
                // Copy observable → series if node is materialised.
                if !slot.output_series_ptr.is_null() {
                    unsafe {
                        (slot.materialize_fn)(
                            slot.output_obs_ptr,
                            slot.output_series_ptr,
                            timestamp,
                        );
                    }
                }
                // Schedule downstream operators.
                for &downstream in &self.edges[slot.output_node_index] {
                    if !self.pending[downstream] {
                        self.pending[downstream] = true;
                        self.heap.push(Reverse(downstream));
                    }
                }
            }
        }
    }
}

impl Default for Scenario {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: Scenario owns all its data through heap-allocated raw pointers
// (Observable, Series, operator states).  Moving the entire Scenario to
// another thread is safe because all pointed-to data is exclusively owned
// and moves with the struct.  This is needed for the PyO3 bridge which
// moves Scenario into py.allow_threads().
unsafe impl Send for Scenario {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators;

    #[test]
    fn simple_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], operators::add());

        unsafe {
            sc.observable_mut(ha).write(&[10.0]);
            sc.observable_mut(hb).write(&[3.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out = unsafe { sc.observable_ref(ho) };
        assert_eq!(out.current_view().as_slice().unwrap(), &[13.0]);
    }

    #[test]
    fn materialized_output() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], operators::add());
        let ho_series = sc.materialize::<f64>(ho);

        unsafe {
            sc.observable_mut(ha).write(&[10.0]);
            sc.observable_mut(hb).write(&[3.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        unsafe {
            sc.observable_mut(ha).write(&[20.0]);
            sc.observable_mut(hb).write(&[7.0]);
        }
        sc.flush(2, &[ha.index, hb.index]);

        // Observable has latest value.
        let obs = unsafe { sc.observable_ref(ho) };
        assert_eq!(obs.current_view().as_slice().unwrap(), &[27.0]);

        // Series has full history (initial + 2 flushes).
        let series = unsafe { sc.series_ref(ho_series) };
        assert_eq!(series.len(), 3);
        assert_eq!(series.index(), &[i64::MIN, 1, 2]);
        assert_eq!(series.values(), &[0.0, 13.0, 27.0]);
    }

    #[test]
    fn chain_operators() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let hab = sc.add_operator([ha, hb], operators::add());
        let hout = sc.add_operator([hab, ha], operators::multiply());

        unsafe {
            sc.observable_mut(ha).write(&[2.0]);
            sc.observable_mut(hb).write(&[3.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out = unsafe { sc.observable_ref(hout) };
        assert_eq!(out.current_view().as_slice().unwrap(), &[10.0]); // (2+3) * 2
    }

    #[test]
    fn sparse_update_skips_inactive() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho1 = sc.add_operator([ha, hb], operators::add());
        let ho1_series = sc.materialize::<f64>(ho1);

        let ho2 = sc.add_source::<f64>(&[], &[0.0]);
        let hc = sc.add_operator([ho2, ha], operators::add());
        let hc_series = sc.materialize::<f64>(hc);

        unsafe {
            sc.observable_mut(ha).write(&[1.0]);
            sc.observable_mut(hb).write(&[2.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out1 = unsafe { sc.series_ref(ho1_series) };
        assert_eq!(out1.len(), 2); // initial + 1 flush
        assert_eq!(out1.current_view().as_slice().unwrap(), &[3.0]);

        // hc produces output (both observables have values), but ho2 is 0.0 (initial).
        let outc = unsafe { sc.series_ref(hc_series) };
        assert_eq!(outc.len(), 2); // initial + 1 flush
        assert_eq!(outc.current_view().as_slice().unwrap(), &[1.0]); // 0.0 + 1.0
    }

    #[test]
    fn incremental_ticks() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], operators::add());
        let ho_series = sc.materialize::<f64>(ho);

        for i in 0..100 {
            let ts = i as i64;
            let va = i as f64;
            let vb = (i * 2) as f64;
            unsafe {
                sc.observable_mut(ha).write(&[va]);
                sc.observable_mut(hb).write(&[vb]);
            }
            sc.flush(ts, &[ha.index, hb.index]);
        }

        let out = unsafe { sc.series_ref(ho_series) };
        assert_eq!(out.len(), 101); // initial + 100 flushes
        assert_eq!(out.current_view().as_slice().unwrap(), &[99.0 + 198.0]);
    }

    #[test]
    fn unmaterialized_intermediate() {
        // Chain: a + b → mid → mid * a → out
        // Only materialize the final output, not mid.
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let hmid = sc.add_operator([ha, hb], operators::add());
        let hout = sc.add_operator([hmid, ha], operators::multiply());
        let hout_series = sc.materialize::<f64>(hout);

        for i in 1..=5 {
            let ts = i as i64;
            let v = i as f64;
            unsafe {
                sc.observable_mut(ha).write(&[v]);
                sc.observable_mut(hb).write(&[v * 2.0]);
            }
            sc.flush(ts, &[ha.index, hb.index]);
        }

        // mid is not materialised — no series exists.
        assert!(sc.nodes[hmid.index].series_ptr.is_null());

        // Final output has history.
        let out = unsafe { sc.series_ref(hout_series) };
        assert_eq!(out.len(), 6); // initial + 5 flushes
        // At tick 5: mid = 5+10 = 15, out = 15*5 = 75
        assert_eq!(out.current_view().as_slice().unwrap(), &[75.0]);
    }

    #[test]
    fn materialize_source() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let ha_series = sc.materialize::<f64>(ha);

        for i in 0..10 {
            unsafe { sc.observable_mut(ha).write(&[i as f64]) };
            sc.flush(i as i64, &[ha.index]);
        }

        let series = unsafe { sc.series_ref(ha_series) };
        assert_eq!(series.len(), 11); // initial + 10 flushes
        assert_eq!(series.current_view().as_slice().unwrap(), &[9.0]);
    }

    // -- Slice operator tests -----------------------------------------------

    #[test]
    fn slice_operator_concat() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[1.0]);
        let hb = sc.add_source::<f64>(&[], &[2.0]);
        let hc = sc.add_source::<f64>(&[], &[3.0]);
        let ho = sc.add_operator([ha, hb, hc], operators::Concat::new(&[], 0));
        let ho_series = sc.materialize::<f64>(ho);

        unsafe {
            sc.observable_mut(ha).write(&[10.0]);
            sc.observable_mut(hb).write(&[20.0]);
            sc.observable_mut(hc).write(&[30.0]);
        }
        sc.flush(1, &[ha.index, hb.index, hc.index]);

        let series = unsafe { sc.series_ref(ho_series) };
        assert_eq!(series.len(), 2); // initial + 1 flush
        assert_eq!(
            series.current_view().as_slice().unwrap(),
            &[10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn slice_operator_stack() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[2], &[1.0, 2.0]);
        let hb = sc.add_source::<f64>(&[2], &[3.0, 4.0]);
        let ho = sc.add_operator([ha, hb], operators::Stack::new(&[2], 0));
        let ho_series = sc.materialize::<f64>(ho);

        sc.flush(1, &[ha.index, hb.index]);

        let series = unsafe { sc.series_ref(ho_series) };
        assert_eq!(series.len(), 2); // initial + 1 flush
        assert_eq!(
            series.current_view().as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn select_operator() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[5], &[10.0, 20.0, 30.0, 40.0, 50.0]);
        let ho = sc.add_operator([ha], operators::Select::flat(vec![1, 3]));
        let ho_series = sc.materialize::<f64>(ho);

        sc.flush(1, &[ha.index]);

        let series = unsafe { sc.series_ref(ho_series) };
        assert_eq!(series.len(), 2); // initial + 1 flush
        assert_eq!(series.current_view().as_slice().unwrap(), &[20.0, 40.0]);
    }

    #[test]
    fn filter_operator() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha], operators::Filter::new(|v: &[f64]| v[0] > 3.0));
        let ho_series = sc.materialize::<f64>(ho);

        // Value 1.0 → filtered out
        unsafe { sc.observable_mut(ha).write(&[1.0]) };
        sc.flush(1, &[ha.index]);
        // Value 5.0 → passes
        unsafe { sc.observable_mut(ha).write(&[5.0]) };
        sc.flush(2, &[ha.index]);
        // Value 2.0 → filtered out
        unsafe { sc.observable_mut(ha).write(&[2.0]) };
        sc.flush(3, &[ha.index]);
        // Value 10.0 → passes
        unsafe { sc.observable_mut(ha).write(&[10.0]) };
        sc.flush(4, &[ha.index]);

        let series = unsafe { sc.series_ref(ho_series) };
        assert_eq!(series.len(), 3); // initial + 2 passes (filtered skipped)
        assert_eq!(series.index(), &[i64::MIN, 2, 4]);
        assert_eq!(series.values(), &[0.0, 5.0, 10.0]);
    }

    #[test]
    fn where_operator() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[3], &[0.0, 0.0, 0.0]);
        let ho = sc.add_operator([ha], operators::Where::new(|v: f64| v > 2.0, 0.0));
        let ho_series = sc.materialize::<f64>(ho);

        unsafe { sc.observable_mut(ha).write(&[1.0, 5.0, 2.0]) };
        sc.flush(1, &[ha.index]);

        let series = unsafe { sc.series_ref(ho_series) };
        assert_eq!(series.len(), 2); // initial + 1 flush
        assert_eq!(series.current_view().as_slice().unwrap(), &[0.0, 5.0, 0.0]);
    }

    #[test]
    fn negate_operator() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha], operators::negate());

        unsafe { sc.observable_mut(ha).write(&[7.0]) };
        sc.flush(1, &[ha.index]);

        let out = unsafe { sc.observable_ref(ho) };
        assert_eq!(out.current_view().as_slice().unwrap(), &[-7.0]);
    }

    // -- run() tests --------------------------------------------------------

    #[tokio::test]
    async fn run_single_source() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let ha_series = sc.materialize::<f64>(ha);

        sc.register_source(
            ha,
            ArraySource::new(vec![1, 2, 3], vec![10.0, 20.0, 30.0], 1),
        );

        sc.run().await;

        let series = unsafe { sc.series_ref(ha_series) };
        assert_eq!(series.len(), 4); // initial + 3 source events
        assert_eq!(series.index(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(series.values(), &[0.0, 10.0, 20.0, 30.0]);
    }

    #[tokio::test]
    async fn run_two_sources_interleaved() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], operators::add());
        let ho_series = sc.materialize::<f64>(ho);

        sc.register_source(ha, ArraySource::new(vec![1, 3], vec![10.0, 30.0], 1));
        sc.register_source(hb, ArraySource::new(vec![2, 3], vec![20.0, 40.0], 1));

        sc.run().await;

        let series = unsafe { sc.series_ref(ho_series) };
        // initial (0.0) + ts=1: a=10,b=0→10 + ts=2: a=10,b=20→30 + ts=3: a=30,b=40→70
        assert_eq!(series.len(), 4);
        assert_eq!(series.index(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(series.values(), &[0.0, 10.0, 30.0, 70.0]);
    }

    #[tokio::test]
    async fn run_coalescing() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], operators::add());
        let ho_series = sc.materialize::<f64>(ho);
        let ha_series = sc.materialize::<f64>(ha);

        sc.register_source(ha, ArraySource::new(vec![1, 2], vec![10.0, 20.0], 1));
        sc.register_source(hb, ArraySource::new(vec![1, 2], vec![100.0, 200.0], 1));

        sc.run().await;

        let series = unsafe { sc.series_ref(ho_series) };
        assert_eq!(series.len(), 3); // initial + 2 coalesced flushes
        assert_eq!(series.index(), &[i64::MIN, 1, 2]);
        assert_eq!(series.values(), &[0.0, 110.0, 220.0]);

        let a_series = unsafe { sc.series_ref(ha_series) };
        assert_eq!(a_series.len(), 3); // initial + 2 source events
    }

    #[tokio::test]
    async fn run_chained_operators() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let hab = sc.add_operator([ha, hb], operators::add());
        let hout = sc.add_operator([hab, ha], operators::multiply());
        let hout_series = sc.materialize::<f64>(hout);

        sc.register_source(ha, ArraySource::new(vec![1, 2], vec![2.0, 5.0], 1));
        sc.register_source(hb, ArraySource::new(vec![1, 2], vec![3.0, 10.0], 1));

        sc.run().await;

        let series = unsafe { sc.series_ref(hout_series) };
        assert_eq!(series.len(), 3); // initial + 2 flushes
        // initial: 0.0, ts=1: (2+3)*2 = 10, ts=2: (5+10)*5 = 75
        assert_eq!(series.values(), &[0.0, 10.0, 75.0]);
    }
}
