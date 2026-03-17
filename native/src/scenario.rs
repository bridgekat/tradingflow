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

use crate::observable::{ObservableHandle, Observable};
use crate::operator::Operator;
use crate::operators::Apply;
use crate::series::{Series, SeriesHandle};

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
    /// Stride of the observable (needed for creating series).
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
    series.append_unchecked(timestamp, obs.last());
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
// Type-erased compute for Apply<T, F>
// ---------------------------------------------------------------------------

/// Universal compute entry point for [`Apply<T, F>`].
///
/// # Safety
///
/// * `input_ptrs` must point to `n_inputs` valid `*mut u8` entries, each
///   actually a `*mut Observable<T>`.
/// * `output_ptr` must point to a valid `Observable<T>`.
/// * `state_ptr` must point to a valid `Apply<T, F>`.
#[inline]
unsafe fn compute_apply<T: Copy, F: Fn(&[&[T]], &mut [T])>(
    timestamp: i64,
    input_ptrs: *const *mut u8,
    n_inputs: usize,
    output_ptr: *mut u8,
    state_ptr: *mut u8,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut Apply<T, F>) };
    let output = unsafe { &mut *(output_ptr as *mut Observable<T>) };
    let inputs: &[&Observable<T>] =
        unsafe { std::slice::from_raw_parts(input_ptrs as *const &Observable<T>, n_inputs) };
    let out = output.vals_mut();
    state.compute(timestamp, inputs, out)
}

/// Drop function for Apply operator slots.
unsafe fn drop_apply<T: Copy, F: Fn(&[&[T]], &mut [T])>(
    state: *mut u8,
    input_ptrs: *const *mut u8,
    n_inputs: usize,
) {
    unsafe {
        drop(Box::from_raw(state as *mut Apply<T, F>));
        drop(Vec::from_raw_parts(
            input_ptrs as *mut *mut u8,
            n_inputs,
            n_inputs,
        ));
    }
}

// ---------------------------------------------------------------------------
// Type-erased compute for heterogeneous operators (via InputTuple)
// ---------------------------------------------------------------------------

/// Generic compute entry point for any operator registered via `add_operator`.
///
/// Monomorphised for each `(I, Op)` pair at registration time.
#[inline]
unsafe fn erased_compute<I, Op>(
    timestamp: i64,
    input_ptrs: *const *mut u8,
    _n_inputs: usize,
    output_obs_ptr: *mut u8,
    state_ptr: *mut u8,
) -> bool
where
    I: crate::input::InputTuple,
    Op: Operator,
    Op::Output: Copy,
    for<'a> Op: Operator<Inputs<'a> = I::Refs<'a>>,
{
    let state = unsafe { &mut *(state_ptr as *mut Op) };
    let output = unsafe { &mut *(output_obs_ptr as *mut Observable<Op::Output>) };
    let inputs = unsafe { I::from_ptrs(input_ptrs) };
    let out = output.vals_mut();
    state.compute(timestamp, inputs, out)
}

/// Drop function for heterogeneous operator slots.
unsafe fn drop_erased_op<Op>(
    state: *mut u8,
    input_ptrs: *const *mut u8,
    n_inputs: usize,
) {
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
// Scenario
// ---------------------------------------------------------------------------

/// Owns all nodes and operators; coordinates DAG execution.
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
        }
    }

    // -- Private: node creation ---------------------------------------------

    /// Internal helper — create a node with an **uninitialised** observable.
    ///
    /// Used for operator output nodes; their initial values are computed
    /// during [`recompute_topo`].
    fn create_node<T: Copy>(&mut self, shape: &[usize]) -> ObservableHandle<T> {
        let idx = self.nodes.len();
        let obs = Box::new(Observable::<T>::new_uninit(shape));
        let stride = obs.stride();
        self.nodes.push(NodeSlot {
            obs_ptr: Box::into_raw(obs) as *mut u8,
            series_ptr: std::ptr::null_mut(),
            materialize_fn: materialize_copy::<T>,
            drop_fn: drop_node::<T>,
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
        let stride = self.nodes[h.index].stride;
        // Reconstruct shape as [stride] (flat).
        let shape = if stride == 1 { vec![] } else { vec![stride] };
        let series = Box::new(Series::<T>::new(&shape));
        self.nodes[h.index].series_ptr = Box::into_raw(series) as *mut u8;
        // Update any existing operator that outputs to this node.
        for slot in &mut self.operators {
            if slot.output_node_index == h.index {
                slot.output_series_ptr = self.nodes[h.index].series_ptr;
            }
        }
        SeriesHandle::new(h.index)
    }

    /// Materialise a node with pre-allocated series capacity.
    pub fn materialize_with_capacity<T: Copy>(
        &mut self,
        h: ObservableHandle<T>,
        cap: usize,
    ) -> SeriesHandle<T> {
        assert!(
            self.nodes[h.index].series_ptr.is_null(),
            "node already materialised"
        );
        let stride = self.nodes[h.index].stride;
        let shape = if stride == 1 { vec![] } else { vec![stride] };
        let series = Box::new(Series::<T>::with_capacity(&shape, cap));
        self.nodes[h.index].series_ptr = Box::into_raw(series) as *mut u8;
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
    pub unsafe fn observable_mut<T: Copy>(
        &mut self,
        h: ObservableHandle<T>,
    ) -> &mut Observable<T> {
        unsafe { &mut *(self.nodes[h.index].obs_ptr as *mut Observable<T>) }
    }

    /// Get a shared reference to the concrete `Observable<T>`.
    #[inline(always)]
    pub unsafe fn observable_ref<T: Copy>(
        &self,
        h: ObservableHandle<T>,
    ) -> &Observable<T> {
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

    // -- Apply registration (homogeneous observable inputs) -----------------

    /// Register an [`Apply`] operator with homogeneous observable inputs.
    ///
    /// Creates the output node internally and returns its handle.
    pub fn add_apply<T: Copy + 'static, F: Fn(&[&[T]], &mut [T]) + 'static>(
        &mut self,
        inputs: &[ObservableHandle<T>],
        apply: Apply<T, F>,
    ) -> ObservableHandle<T> {
        // Determine output shape from inputs (all must have same stride).
        let shape: &[usize] = if !inputs.is_empty() {
            let stride = self.nodes[inputs[0].index].stride;
            if stride == 1 { &[] } else { &[stride] }
        } else {
            &[]
        };
        let output = self.create_node::<T>(shape);
        let op_idx = self.operators.len();

        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(inputs.len());
        for h in inputs {
            self.edges[h.index].push(op_idx);
            ptrs.push(self.nodes[h.index].obs_ptr);
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
            compute_fn: compute_apply::<T, F>,
            state: Box::into_raw(Box::new(apply)) as *mut u8,
            drop_fn: drop_apply::<T, F>,
        });
        self.topo_dirty = true;
        output
    }

    // -- Generic operator registration (heterogeneous inputs) ---------------

    /// Register an operator with heterogeneous inputs specified by an
    /// [`InputTuple`].
    ///
    /// Creates the output node internally and returns its handle.
    pub fn add_operator<I, Op>(
        &mut self,
        inputs: I::Handles,
        output_shape: &[usize],
        op: Op,
    ) -> ObservableHandle<Op::Output>
    where
        I: crate::input::InputTuple + 'static,
        Op: Operator + 'static,
        Op::Output: Copy,
        for<'a> Op: Operator<Inputs<'a> = I::Refs<'a>>,
    {
        let output = self.create_node::<Op::Output>(output_shape);
        let op_idx = self.operators.len();

        let ptrs = I::extract_ptrs(&self.nodes, inputs);
        let n_inputs = ptrs.len();
        let input_ptrs = ptrs.as_ptr();
        std::mem::forget(ptrs);

        // Register edges from input nodes to this operator.
        for idx in I::node_indices(inputs) {
            self.edges[idx].push(op_idx);
        }

        self.operators.push(OperatorSlot {
            output_node_index: output.index,
            input_ptrs,
            n_inputs,
            output_obs_ptr: self.nodes[output.index].obs_ptr,
            output_series_ptr: self.nodes[output.index].series_ptr,
            materialize_fn: self.nodes[output.index].materialize_fn,
            compute_fn: erased_compute::<I, Op>,
            state: Box::into_raw(Box::new(op)) as *mut u8,
            drop_fn: drop_erased_op::<Op>,
        });
        self.topo_dirty = true;
        output
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
        let ho = sc.add_apply(&[ha, hb], operators::add());

        unsafe {
            sc.observable_mut(ha).write(&[10.0]);
            sc.observable_mut(hb).write(&[3.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out = unsafe { sc.observable_ref(ho) };
        assert_eq!(out.last(), &[13.0]);
    }

    #[test]
    fn materialized_output() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_apply(&[ha, hb], operators::add());
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
        assert_eq!(obs.last(), &[27.0]);

        // Series has full history.
        let series = unsafe { sc.series_ref(ho_series) };
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[1, 2]);
        assert_eq!(series.values(), &[13.0, 27.0]);
    }

    #[test]
    fn chain_operators() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let hab = sc.add_apply(&[ha, hb], operators::add());
        let hout = sc.add_apply(&[hab, ha], operators::multiply());

        unsafe {
            sc.observable_mut(ha).write(&[2.0]);
            sc.observable_mut(hb).write(&[3.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out = unsafe { sc.observable_ref(hout) };
        assert_eq!(out.last(), &[10.0]); // (2+3) * 2
    }

    #[test]
    fn sparse_update_skips_inactive() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho1 = sc.add_apply(&[ha, hb], operators::add());
        let ho1_series = sc.materialize::<f64>(ho1);

        // op1: ho2 is never written to → this op uses it as input
        let ho2 = sc.add_source::<f64>(&[], &[0.0]);
        let hc = sc.add_apply(&[ho2, ha], operators::add());
        let hc_series = sc.materialize::<f64>(hc);

        unsafe {
            sc.observable_mut(ha).write(&[1.0]);
            sc.observable_mut(hb).write(&[2.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out1 = unsafe { sc.series_ref(ho1_series) };
        assert_eq!(out1.len(), 1);
        assert_eq!(out1.last(), &[3.0]);

        // hc produces output (both observables have values), but ho2 is 0.0 (initial).
        let outc = unsafe { sc.series_ref(hc_series) };
        assert_eq!(outc.len(), 1);
        assert_eq!(outc.last(), &[1.0]); // 0.0 + 1.0
    }

    #[test]
    fn incremental_ticks() {
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let ho = sc.add_apply(&[ha, hb], operators::add());
        let ho_series = sc.materialize_with_capacity::<f64>(ho, 100);

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
        assert_eq!(out.len(), 100);
        assert_eq!(out.last(), &[99.0 + 198.0]);
    }

    #[test]
    fn unmaterialized_intermediate() {
        // Chain: a + b → mid → mid * a → out
        // Only materialize the final output, not mid.
        let mut sc = Scenario::new();
        let ha = sc.add_source::<f64>(&[], &[0.0]);
        let hb = sc.add_source::<f64>(&[], &[0.0]);
        let hmid = sc.add_apply(&[ha, hb], operators::add());
        let hout = sc.add_apply(&[hmid, ha], operators::multiply());
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
        assert_eq!(out.len(), 5);
        // At tick 5: mid = 5+10 = 15, out = 15*5 = 75
        assert_eq!(out.last(), &[75.0]);
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
        assert_eq!(series.len(), 10);
        assert_eq!(series.last(), &[9.0]);
        assert_eq!(series.timestamps(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
