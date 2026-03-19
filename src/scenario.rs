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
//! in topological order.  This is O(active_operators) not O(total_operators).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use crate::observable::Observable;
use crate::operator::{InputRefs, Operator};
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
    input_ptrs: *const *mut u8,
    n_inputs: usize,
    output_obs_ptr: *mut u8,
    output_series_ptr: *mut u8,
    materialize_fn: unsafe fn(*mut u8, *mut u8, i64),
    compute_fn: unsafe fn(i64, *const *mut u8, usize, *mut u8, *mut u8) -> bool,
    state: *mut u8,
    drop_fn: unsafe fn(*mut u8, *const *mut u8, usize),
}

impl Drop for OperatorSlot {
    fn drop(&mut self) {
        unsafe { (self.drop_fn)(self.state, self.input_ptrs, self.n_inputs) }
    }
}

// ---------------------------------------------------------------------------
// Type-erased compute for slice-input operators (&[S::Ref<'_>])
// ---------------------------------------------------------------------------

/// Type-erased compute entry point, monomorphised for each `Op`.
///
/// Uses [`InputRefs::from_ptrs`] to reconstruct the typed references.
#[inline]
unsafe fn erased_compute<Op>(
    timestamp: i64,
    input_ptrs: *const *mut u8,
    n_inputs: usize,
    output_obs_ptr: *mut u8,
    state_ptr: *mut u8,
) -> bool
where
    Op: Operator,
    Op::Scalar: Copy,
    for<'a> Op::Inputs<'a>: crate::operator::InputRefs<'a>,
{
    let state = unsafe { &mut *(state_ptr as *mut Op) };
    let output = unsafe { &mut *(output_obs_ptr as *mut Observable<Op::Scalar>) };
    let inputs = unsafe { Op::Inputs::from_raw(input_ptrs as *const *const u8, n_inputs) };
    let out = output.current_mut();
    state.compute(timestamp, inputs, out)
}

/// Drop function for heterogeneous operator slots.
unsafe fn drop_erased_op<Op>(state: *mut u8, input_ptrs: *const *mut u8, n_inputs: usize) {
    unsafe {
        drop(Box::from_raw(state as *mut Op));
        drop(Vec::from_raw_parts(
            input_ptrs as *mut *mut u8,
            n_inputs,
            n_inputs,
        ));
    }
}

// ---------------------------------------------------------------------------
// SourceSlot (type-erased)
// ---------------------------------------------------------------------------

/// Type-erased source: stores a boxed Source and the information needed to
/// write yielded values into the correct observable.
struct SourceSlot {
    node_index: usize,
    source: Box<dyn Source>,
    element_size: usize, // stride * size_of::<T>() in bytes
    /// Type-erased: write raw bytes into the observable at obs_ptr.
    write_fn: unsafe fn(*mut u8, &[u8]),
}

/// Typed write function: reinterpret bytes as `&[T]` and write to Observable.
unsafe fn write_source_typed<T: Copy>(obs_ptr: *mut u8, bytes: &[u8]) {
    let obs = unsafe { &mut *(obs_ptr as *mut Observable<T>) };
    let values = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const T,
            bytes.len() / std::mem::size_of::<T>(),
        )
    };
    obs.write(values);
}

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

/// Owns all nodes, operators, and sources; coordinates DAG execution.
pub struct Scenario {
    nodes: Vec<NodeSlot>,
    /// `edges[node_idx]` → operator indices that read from this node.
    edges: Vec<Vec<usize>>,
    operators: Vec<OperatorSlot>,
    topo_order: Vec<usize>,
    topo_rank: Vec<usize>,
    topo_dirty: bool,
    // Reusable per-flush scratch space:
    pending: Vec<bool>,
    heap: BinaryHeap<Reverse<(usize, usize)>>,
    // Sources for run():
    source_slots: Vec<SourceSlot>,
}

impl Scenario {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            operators: Vec::new(),
            topo_order: Vec::new(),
            topo_rank: Vec::new(),
            topo_dirty: true,
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
        self.topo_dirty = true;
        ObservableHandle::new(idx)
    }

    // -- Source registration ------------------------------------------------

    /// Register a source node with an explicit initial value.
    ///
    /// The initial value is written to the observable immediately.  Operator
    /// outputs are initialised later during [`recompute_topo`].
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
        self.topo_dirty = true;
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

    /// Register an operator with inputs specified as a slice of [`InputSlot`]
    /// handles.
    ///
    /// `S` is the handle type (e.g. `ObservableHandle<T>` or `SeriesHandle<T>`).
    /// The operator's `Inputs<'a>` must be reconstructible from raw pointers
    /// via [`InputRefs`].
    ///
    /// Creates the output node internally and returns its handle.
    pub fn add_operator<'a, Op>(
        &'a mut self,
        inputs: impl Into<<Op::Inputs<'a> as InputRefs<'a>>::Handles>,
        initial: &[Op::Scalar],
        op: Op,
    ) -> ObservableHandle<Op::Scalar>
    where
        Op: Operator,
    {
        let handles = inputs.into();
        let node_ids = <Op::Inputs<'_> as InputRefs<'_>>::node_ids(&handles);
        let input_shapes: Box<[&[usize]]> = node_ids
            .iter()
            .map(|&(i, _)| &*self.nodes[i].shape)
            .collect();
        let output_shape = op.output_shape(&input_shapes);
        let output = self.create_node::<Op::Scalar>(&output_shape, initial);
        let op_idx = self.operators.len();

        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(node_ids.len());
        for &(i, is_series) in node_ids.iter() {
            self.edges[i].push(op_idx);
            if is_series {
                ptrs.push(self.nodes[i].series_ptr);
            } else {
                ptrs.push(self.nodes[i].obs_ptr);
            }
        }
        let n_inputs = ptrs.len();
        let input_ptrs = ptrs.as_ptr();
        std::mem::forget(ptrs);

        self.operators.push(OperatorSlot {
            output_node_index: output.index,
            input_ptrs,
            n_inputs,
            output_obs_ptr: self.nodes[output.index].obs_ptr,
            output_series_ptr: self.nodes[output.index].series_ptr,
            materialize_fn: self.nodes[output.index].materialize_fn,
            compute_fn: erased_compute::<Op>,
            state: Box::into_raw(Box::new(op)) as *mut u8,
            drop_fn: drop_erased_op::<Op>,
        });
        self.topo_dirty = true;
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
        compute_fn: unsafe fn(i64, *const *mut u8, usize, *mut u8, *mut u8) -> bool,
        state: *mut u8,
        drop_fn: unsafe fn(*mut u8, *const *mut u8, usize),
    ) {
        let op_idx = self.operators.len();

        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(input_indices.len());
        for &idx in input_indices {
            self.edges[idx].push(op_idx);
            ptrs.push(self.nodes[idx].obs_ptr);
        }
        let n_inputs = ptrs.len();
        let input_ptrs = ptrs.as_ptr();
        std::mem::forget(ptrs);

        self.operators.push(OperatorSlot {
            output_node_index,
            input_ptrs,
            n_inputs,
            output_obs_ptr: self.nodes[output_node_index].obs_ptr,
            output_series_ptr: self.nodes[output_node_index].series_ptr,
            materialize_fn: self.nodes[output_node_index].materialize_fn,
            compute_fn,
            state,
            drop_fn,
        });
        self.topo_dirty = true;
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
    pub(crate) fn register_source_typed<T: Copy + 'static>(
        &mut self,
        node_index: usize,
        source: Box<dyn Source>,
    ) {
        let stride = self.nodes[node_index].stride;
        self.source_slots.push(SourceSlot {
            node_index,
            source,
            element_size: stride * std::mem::size_of::<T>(),
            write_fn: write_source_typed::<T>,
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
    pub fn register_source<T: Copy + 'static>(
        &mut self,
        handle: ObservableHandle<T>,
        source: Box<dyn Source>,
    ) {
        let stride = self.nodes[handle.index].stride;
        self.source_slots.push(SourceSlot {
            node_index: handle.index,
            source,
            element_size: stride * std::mem::size_of::<T>(),
            write_fn: write_source_typed::<T>,
        });
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
        self.recompute_topo();

        // Subscribe all sources and build per-source runtime state.
        let slots: Vec<SourceSlot> = self.source_slots.drain(..).collect();
        let mut states: Vec<RunSourceState> = Vec::with_capacity(slots.len());
        for slot in slots {
            let (hist, live) = slot.source.subscribe().await;
            states.push(RunSourceState {
                node_index: slot.node_index,
                element_size: slot.element_size,
                write_fn: slot.write_fn,
                obs_ptr: self.nodes[slot.node_index].obs_ptr,
                hist_iter: Some(hist),
                live_iter: Some(live),
                buf: vec![0u8; slot.element_size],
                pending_ts: None,
                hist_exhausted: false,
                live_exhausted: false,
            });
        }

        // Fetch initial historical event from each source.
        for st in &mut states {
            st.fetch_hist().await;
        }

        // POCQ state.
        let mut queue_ts: Option<i64> = None;
        let mut queue_sources: Vec<usize> = Vec::new(); // node indices

        loop {
            // Check if all historical sources have pending events (or are exhausted).
            let hist_blocked = states
                .iter()
                .any(|st| st.hist_iter.is_some() && st.pending_ts.is_none());
            if hist_blocked {
                // Should not happen in a well-formed sequential loop since
                // fetch_hist() is awaited above. But for live sources that
                // haven't been fetched yet, we'd await here.
                // For now, fetch any source that lacks a pending event.
                for st in &mut states {
                    if st.hist_iter.is_some() && st.pending_ts.is_none() {
                        st.fetch_hist().await;
                    }
                    if st.live_iter.is_some() && !st.live_exhausted && st.pending_ts.is_none() {
                        st.fetch_live().await;
                    }
                }
            }

            // Find the minimum pending timestamp across all sources.
            let min_ts = states.iter().filter_map(|st| st.pending_ts).min();

            let Some(min_ts) = min_ts else {
                // All sources exhausted — flush remaining queue and exit.
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
            for st in &mut states {
                if st.pending_ts == Some(min_ts) {
                    // Write the buffered value to the observable.
                    unsafe { (st.write_fn)(st.obs_ptr, &st.buf) };
                    queue_sources.push(st.node_index);
                    st.pending_ts = None;

                    // Fetch next event from this source.
                    if st.hist_iter.is_some() {
                        st.fetch_hist().await;
                    } else if st.live_iter.is_some() {
                        st.fetch_live().await;
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
        if self.topo_dirty {
            self.recompute_topo();
        }
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
                    self.heap.push(Reverse((self.topo_rank[op_idx], op_idx)));
                }
            }
        }
        // Process in topological order; propagate downstream on produce.
        while let Some(Reverse((_, op_idx))) = self.heap.pop() {
            self.pending[op_idx] = false;
            let slot = &self.operators[op_idx];
            // SAFETY: pointers were set up correctly at registration time.
            let produced = unsafe {
                (slot.compute_fn)(
                    timestamp,
                    slot.input_ptrs,
                    slot.n_inputs,
                    slot.output_obs_ptr,
                    slot.state,
                )
            };
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
                        self.heap
                            .push(Reverse((self.topo_rank[downstream], downstream)));
                    }
                }
            }
        }
    }

    // -- Topology -----------------------------------------------------------

    fn recompute_topo(&mut self) {
        let n = self.operators.len();
        // Build operator→operator edge list from node→operator edges.
        let mut op_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, slot) in self.operators.iter().enumerate() {
            for &d in &self.edges[slot.output_node_index] {
                op_edges[i].push(d);
            }
        }
        // DFS topological sort with cycle detection.
        #[derive(Clone, Copy, PartialEq)]
        enum Color {
            White,
            Grey,
            Black,
        }
        let mut color = vec![Color::White; n];
        let mut order = Vec::with_capacity(n);

        fn dfs(u: usize, edges: &[Vec<usize>], color: &mut [Color], order: &mut Vec<usize>) {
            if color[u] == Color::Black {
                return;
            }
            assert!(color[u] != Color::Grey, "cycle in operator graph");
            color[u] = Color::Grey;
            for &v in &edges[u] {
                dfs(v, edges, color, order);
            }
            color[u] = Color::Black;
            order.push(u);
        }

        for i in 0..n {
            if color[i] == Color::White {
                dfs(i, &op_edges, &mut color, &mut order);
            }
        }
        order.reverse();

        // Build rank (inverse of order) for heap ordering.
        self.topo_rank.resize(n, 0);
        for (rank, &op_idx) in order.iter().enumerate() {
            self.topo_rank[op_idx] = rank;
        }
        self.topo_order = order;
        self.pending.resize(n, false);

        // Initialise operator outputs by running compute once in topo order.
        // Source observables already hold their initial values (set by add_source).
        // Each operator reads initialised inputs and writes its output observable.
        // Series are NOT appended to — only observables are initialised.
        for &op_idx in &self.topo_order {
            let slot = &self.operators[op_idx];
            unsafe {
                (slot.compute_fn)(
                    0, // dummy timestamp — init only
                    slot.input_ptrs,
                    slot.n_inputs,
                    slot.output_obs_ptr,
                    slot.state,
                );
            }
        }

        self.topo_dirty = false;
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
// RunSourceState — per-source mutable state during run()
// ---------------------------------------------------------------------------

/// Per-source runtime state used by [`Scenario::run`].
struct RunSourceState {
    node_index: usize,
    #[allow(dead_code)]
    element_size: usize,
    write_fn: unsafe fn(*mut u8, &[u8]),
    obs_ptr: *mut u8,
    hist_iter: Option<Box<dyn crate::source::HistoricalIter>>,
    live_iter: Option<Box<dyn crate::source::LiveIter>>,
    buf: Vec<u8>,
    /// Pending timestamp (None = no pending event).
    pending_ts: Option<i64>,
    hist_exhausted: bool,
    live_exhausted: bool,
}

impl RunSourceState {
    /// Fetch the next historical event into the buffer.
    async fn fetch_hist(&mut self) {
        if let Some(ref mut hist) = self.hist_iter {
            match hist.next_into(&mut self.buf).await {
                Some(ts) => {
                    self.pending_ts = Some(ts);
                }
                None => {
                    self.hist_iter = None;
                    self.hist_exhausted = true;
                    // Try live if available.
                    self.fetch_live().await;
                }
            }
        }
    }

    /// Fetch the next live event into the buffer.
    async fn fetch_live(&mut self) {
        if let Some(ref mut live) = self.live_iter {
            if live.next_into(&mut self.buf).await {
                // For live events, use wall-clock time.
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as i64;
                self.pending_ts = Some(ts);
            } else {
                self.live_iter = None;
                self.live_exhausted = true;
            }
        }
    }
}

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
        let ho = sc.add_operator([ha, hb], &[0.0], operators::add());

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
        let ho = sc.add_operator([ha, hb], &[0.0], operators::add());
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
        let hab = sc.add_operator([ha, hb], &[0.0], operators::add());
        let hout = sc.add_operator([hab, ha], &[0.0], operators::multiply());

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
        let ho1 = sc.add_operator([ha, hb], &[0.0], operators::add());
        let ho1_series = sc.materialize::<f64>(ho1);

        let ho2 = sc.add_source::<f64>(&[], &[0.0]);
        let hc = sc.add_operator([ho2, ha], &[0.0], operators::add());
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
        let ho = sc.add_operator([ha, hb], &[0.0], operators::add());
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
        let hmid = sc.add_operator([ha, hb], &[0.0], operators::add());
        let hout = sc.add_operator([hmid, ha], &[0.0], operators::multiply());
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
        let ho = sc.add_operator(
            [ha, hb, hc],
            &[0.0, 0.0, 0.0],
            operators::Concat::new(&[], 0),
        );
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
        let ho = sc.add_operator(
            [ha, hb],
            &[0.0, 0.0, 0.0, 0.0],
            operators::Stack::new(&[2], 0),
        );
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
        let ho = sc.add_operator([ha], &[0.0, 0.0], operators::Select::flat(vec![1, 3]));
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
        let ho = sc.add_operator([ha], &[0.0], operators::Filter::new(|v: &[f64]| v[0] > 3.0));
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
        let ho = sc.add_operator(
            [ha],
            &[0.0, 0.0, 0.0],
            operators::Where::new(|v: f64| v > 2.0, 0.0),
        );
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
        let ho = sc.add_operator([ha], &[0.0], operators::negate());

        unsafe { sc.observable_mut(ha).write(&[7.0]) };
        sc.flush(1, &[ha.index]);

        let out = unsafe { sc.observable_ref(ho) };
        assert_eq!(out.current_view().as_slice().unwrap(), &[-7.0]);
    }

    // -- run() tests --------------------------------------------------------

    #[tokio::test]
    async fn run_single_source() {
        use crate::source::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let ha_series = sc.materialize::<f64>(ha);

        sc.register_source(
            ha,
            Box::new(ArraySource::new(vec![1, 2, 3], vec![10.0, 20.0, 30.0], 1)),
        );

        sc.run().await;

        let series = unsafe { sc.series_ref(ha_series) };
        assert_eq!(series.len(), 4); // initial + 3 source events
        assert_eq!(series.index(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(series.values(), &[0.0, 10.0, 20.0, 30.0]);
    }

    #[tokio::test]
    async fn run_two_sources_interleaved() {
        use crate::source::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], &[0.0], operators::add());
        let ho_series = sc.materialize::<f64>(ho);

        sc.register_source(
            ha,
            Box::new(ArraySource::new(vec![1, 3], vec![10.0, 30.0], 1)),
        );
        sc.register_source(
            hb,
            Box::new(ArraySource::new(vec![2, 3], vec![20.0, 40.0], 1)),
        );

        sc.run().await;

        let series = unsafe { sc.series_ref(ho_series) };
        // initial (0.0) + ts=1: a=10,b=0→10 + ts=2: a=10,b=20→30 + ts=3: a=30,b=40→70
        assert_eq!(series.len(), 4);
        assert_eq!(series.index(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(series.values(), &[0.0, 10.0, 30.0, 70.0]);
    }

    #[tokio::test]
    async fn run_coalescing() {
        use crate::source::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], &[0.0], operators::add());
        let ho_series = sc.materialize::<f64>(ho);
        let ha_series = sc.materialize::<f64>(ha);

        sc.register_source(
            ha,
            Box::new(ArraySource::new(vec![1, 2], vec![10.0, 20.0], 1)),
        );
        sc.register_source(
            hb,
            Box::new(ArraySource::new(vec![1, 2], vec![100.0, 200.0], 1)),
        );

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
        use crate::source::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let hab = sc.add_operator([ha, hb], &[0.0], operators::add());
        let hout = sc.add_operator([hab, ha], &[0.0], operators::multiply());
        let hout_series = sc.materialize::<f64>(hout);

        sc.register_source(
            ha,
            Box::new(ArraySource::new(vec![1, 2], vec![2.0, 5.0], 1)),
        );
        sc.register_source(
            hb,
            Box::new(ArraySource::new(vec![1, 2], vec![3.0, 10.0], 1)),
        );

        sc.run().await;

        let series = unsafe { sc.series_ref(hout_series) };
        assert_eq!(series.len(), 3); // initial + 2 flushes
        // initial: 0.0, ts=1: (2+3)*2 = 10, ts=2: (5+10)*5 = 75
        assert_eq!(series.values(), &[0.0, 10.0, 75.0]);
    }
}
