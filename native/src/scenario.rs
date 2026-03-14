//! Scenario — the DAG runtime that owns series and dispatches operators.
//!
//! # Architecture
//!
//! All series are stored as type-erased [`ErasedSeries`] in a flat `Vec`.
//! Each operator is wrapped in an [`OperatorSlot`] that stores:
//!
//! * Pre-cast input/output raw pointers (computed once at registration).
//! * A uniform `compute_fn` function pointer — **not** a closure — so the
//!   slot layout is fixed and there is no per-call dynamic dispatch overhead.
//! * The operator state as `*mut u8` (the implementing struct is the state).
//!
//! Type safety is enforced at registration time via [`SeriesHandle<T>`]
//! generics.  After registration the scenario operates on raw pointers only.
//!
//! # Flush algorithm
//!
//! On each tick the caller appends to source series and calls [`flush`].
//! A min-heap processes only the operators reachable from the updated sources,
//! in topological order.  This is O(active_operators) not O(total_operators),
//! which matters for large sparse graphs.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::operator::Operator;
use crate::operators::Apply;
use crate::series::{ErasedSeries, Series, SeriesHandle};

// ---------------------------------------------------------------------------
// OperatorSlot (type-erased)
// ---------------------------------------------------------------------------

/// Type-erased operator with pre-computed I/O pointers.
///
/// The `compute_fn` signature is the universal calling convention for all
/// operators in the scenario:
///
/// ```text
/// unsafe fn(timestamp, input_ptrs, n_inputs, output_ptr, state_ptr) -> bool
/// ```
pub(crate) struct OperatorSlot {
    pub(crate) output_index: usize,
    input_ptrs: *const *mut u8,
    n_inputs: usize,
    output_ptr: *mut u8,
    compute_fn: unsafe fn(i64, *const *mut u8, usize, *mut u8, *mut u8) -> bool,
    state: *mut u8,
    drop_fn: unsafe fn(*mut u8, *const *mut u8, usize),
}

impl Drop for OperatorSlot {
    fn drop(&mut self) {
        // SAFETY: `drop_fn` knows the concrete types for `state` and `input_ptrs`.
        unsafe { (self.drop_fn)(self.state, self.input_ptrs, self.n_inputs) }
    }
}

// ---------------------------------------------------------------------------
// Type-erased compute function for Apply<T, F>
// ---------------------------------------------------------------------------

/// Universal compute entry point for [`Apply<T, F>`].
///
/// # Safety
///
/// * `input_ptrs` must point to `n_inputs` valid `*mut u8` entries, each
///   actually a `*mut Series<T>`.
/// * `output_ptr` must point to a valid `Series<T>`.
/// * `state_ptr` must point to a valid `Apply<T, F>`.
///
/// These invariants are established at registration time by [`Scenario::add_apply`].
#[inline]
unsafe fn compute_apply<T: Copy + Default, F: Fn(&[&[T]], &mut [T])>(
    timestamp: i64,
    input_ptrs: *const *mut u8,
    n_inputs: usize,
    output_ptr: *mut u8,
    state_ptr: *mut u8,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut Apply<T, F>) };
    let output = unsafe { &mut *(output_ptr as *mut Series<T>) };
    // SAFETY: `*mut u8` and `&Series<T>` are both pointer-width; the underlying
    // objects are `Series<T>` as ensured by registration.
    let inputs: &[&Series<T>] =
        unsafe { std::slice::from_raw_parts(input_ptrs as *const &Series<T>, n_inputs) };
    let out = output.reserve_slot();
    if state.compute(timestamp, inputs, out) {
        output.commit(timestamp);
        true
    } else {
        false
    }
}

/// Drop function that knows the concrete types.
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
// Scenario
// ---------------------------------------------------------------------------

/// Owns all series and operators; coordinates DAG execution.
pub struct Scenario {
    series: Vec<ErasedSeries>,
    /// `edges[series_idx]` → operator indices that read from this series.
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
            series: Vec::new(),
            edges: Vec::new(),
            operators: Vec::new(),
            topo_order: Vec::new(),
            topo_rank: Vec::new(),
            topo_dirty: true,
            pending: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    // -- Series registration ------------------------------------------------

    /// Register a new series with default capacity.
    pub fn add_series<T: Copy>(&mut self, shape: &[usize]) -> SeriesHandle<T> {
        let idx = self.series.len();
        self.series.push(ErasedSeries::new::<T>(shape));
        self.edges.push(Vec::new());
        self.topo_dirty = true;
        SeriesHandle::new(idx)
    }

    /// Register a new series with pre-allocated capacity.
    pub fn add_series_with_capacity<T: Copy>(
        &mut self,
        shape: &[usize],
        cap: usize,
    ) -> SeriesHandle<T> {
        let idx = self.series.len();
        self.series
            .push(ErasedSeries::with_capacity::<T>(shape, cap));
        self.edges.push(Vec::new());
        self.topo_dirty = true;
        SeriesHandle::new(idx)
    }

    // -- Series access (unsafe: caller must use matching T) -----------------

    /// Get a mutable reference to the concrete `Series<T>` behind a handle.
    ///
    /// # Safety
    ///
    /// The handle must have been created by this scenario with the same `T`.
    #[inline(always)]
    pub unsafe fn series_mut<T: Copy>(&mut self, h: SeriesHandle<T>) -> &mut Series<T> {
        unsafe { &mut *(self.series[h.index].ptr as *mut Series<T>) }
    }

    /// Get a shared reference to the concrete `Series<T>` behind a handle.
    ///
    /// # Safety
    ///
    /// The handle must have been created by this scenario with the same `T`.
    #[inline(always)]
    pub unsafe fn series_ref<T: Copy>(&self, h: SeriesHandle<T>) -> &Series<T> {
        unsafe { &*(self.series[h.index].ptr as *const Series<T>) }
    }

    // -- Operator registration ----------------------------------------------

    /// Register an [`Apply`] operator.
    ///
    /// Type safety: the generic parameters ensure that `inputs` and `output`
    /// all refer to `Series<T>`.  After this call the concrete types are
    /// forgotten and the operator runs through type-erased function pointers.
    pub fn add_apply<T: Copy + Default + 'static, F: Fn(&[&[T]], &mut [T]) + 'static>(
        &mut self,
        inputs: &[SeriesHandle<T>],
        output: SeriesHandle<T>,
        apply: Apply<T, F>,
    ) {
        let op_idx = self.operators.len();
        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(inputs.len());
        for h in inputs {
            self.edges[h.index].push(op_idx);
            ptrs.push(self.series[h.index].ptr);
        }
        let n_inputs = ptrs.len();
        let input_ptrs = ptrs.as_ptr();
        std::mem::forget(ptrs);

        self.operators.push(OperatorSlot {
            output_index: output.index,
            input_ptrs,
            n_inputs,
            output_ptr: self.series[output.index].ptr,
            compute_fn: compute_apply::<T, F>,
            state: Box::into_raw(Box::new(apply)) as *mut u8,
            drop_fn: drop_apply::<T, F>,
        });
        self.topo_dirty = true;
    }

    // -- Execution ----------------------------------------------------------

    /// Propagate updates through the DAG.
    ///
    /// After appending to source series, call this with the indices of the
    /// updated series.  Only operators reachable from those series will
    /// execute, in topological order.
    #[inline]
    pub fn flush(&mut self, timestamp: i64, updated_sources: &[usize]) {
        if self.topo_dirty {
            self.recompute_topo();
        }
        // Seed the min-heap with operators directly downstream of updated series.
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
                    slot.output_ptr,
                    slot.state,
                )
            };
            if produced {
                for &downstream in &self.edges[slot.output_index] {
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
        // Build operator→operator edge list from series→operator edges.
        let mut op_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, slot) in self.operators.iter().enumerate() {
            for &d in &self.edges[slot.output_index] {
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
        let ha: SeriesHandle<f64> = sc.add_series(&[]);
        let hb: SeriesHandle<f64> = sc.add_series(&[]);
        let ho: SeriesHandle<f64> = sc.add_series(&[]);
        sc.add_apply(&[ha, hb], ho, operators::add());

        unsafe {
            sc.series_mut(ha).append_unchecked(1, &[10.0]);
            sc.series_mut(hb).append_unchecked(1, &[3.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out = unsafe { sc.series_ref(ho) };
        assert_eq!(out.len(), 1);
        assert_eq!(out.last(), &[13.0]);
    }

    #[test]
    fn chain_operators() {
        let mut sc = Scenario::new();
        let ha: SeriesHandle<f64> = sc.add_series(&[]);
        let hb: SeriesHandle<f64> = sc.add_series(&[]);
        let hab: SeriesHandle<f64> = sc.add_series(&[]);
        let hout: SeriesHandle<f64> = sc.add_series(&[]);
        sc.add_apply(&[ha, hb], hab, operators::add());
        sc.add_apply(&[hab, ha], hout, operators::multiply());

        unsafe {
            sc.series_mut(ha).append_unchecked(1, &[2.0]);
            sc.series_mut(hb).append_unchecked(1, &[3.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out = unsafe { sc.series_ref(hout) };
        assert_eq!(out.len(), 1);
        assert_eq!(out.last(), &[10.0]); // (2+3) * 2
    }

    #[test]
    fn sparse_update_skips_inactive() {
        let mut sc = Scenario::new();
        let ha: SeriesHandle<f64> = sc.add_series(&[]);
        let hb: SeriesHandle<f64> = sc.add_series(&[]);
        let ho1: SeriesHandle<f64> = sc.add_series(&[]);
        let ho2: SeriesHandle<f64> = sc.add_series(&[]);
        // op0: ha + hb → ho1 (will fire)
        sc.add_apply(&[ha, hb], ho1, operators::add());
        // op1: ho2 is never written → this op should never fire
        let hc: SeriesHandle<f64> = sc.add_series(&[]);
        sc.add_apply(&[ho2, ha], hc, operators::add());

        unsafe {
            sc.series_mut(ha).append_unchecked(1, &[1.0]);
            sc.series_mut(hb).append_unchecked(1, &[2.0]);
        }
        sc.flush(1, &[ha.index, hb.index]);

        let out1 = unsafe { sc.series_ref(ho1) };
        assert_eq!(out1.len(), 1);
        assert_eq!(out1.last(), &[3.0]);

        // hc should have no output (ho2 was never updated → op1 fires because
        // ha updated, but ho2 is empty → compute returns false).
        let outc = unsafe { sc.series_ref(hc) };
        assert_eq!(outc.len(), 0);
    }

    #[test]
    fn incremental_ticks() {
        let mut sc = Scenario::new();
        let ha: SeriesHandle<f64> = sc.add_series(&[]);
        let hb: SeriesHandle<f64> = sc.add_series(&[]);
        let ho: SeriesHandle<f64> = sc.add_series(&[]);
        sc.add_apply(&[ha, hb], ho, operators::add());

        for i in 0..100 {
            let ts = i as i64;
            let va = i as f64;
            let vb = (i * 2) as f64;
            unsafe {
                sc.series_mut(ha).append_unchecked(ts, &[va]);
                sc.series_mut(hb).append_unchecked(ts, &[vb]);
            }
            sc.flush(ts, &[ha.index, hb.index]);
        }

        let out = unsafe { sc.series_ref(ho) };
        assert_eq!(out.len(), 100);
        assert_eq!(out.last(), &[99.0 + 198.0]);
    }
}
