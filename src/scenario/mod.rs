//! Scenario — the DAG runtime for event-driven computation.
//!
//! A [`Scenario`] is a directed acyclic graph of nodes, where each node holds
//! a [`Store`](crate::store::Store).  Nodes are fed by [`Source`](crate::source::Source)s
//! and connected by [`Operator`](crate::operator::Operator)s.
//!
//! # Architecture
//!
//! Internally, nodes are stored as type-erased `(pointer, TypeId)` slots.
//! Type safety is enforced at registration time via [`Handle<T>`] and
//! [`TypeId`] checks.  After registration, operator dispatch uses raw pointer
//! casts through monomorphised function pointers — zero dynamic dispatch
//! overhead on the hot path.
//!
//! # Submodules
//!
//! * [`handle`] — [`Handle<T>`], [`InputKindsHandles`] (typed interface).
//! * [`graph`] — [`Graph`], topological flush.
//! * [`node`] — [`Node`], [`Closure`], typed construction helpers.
//! * [`runner`] — [`SourceState`](runner::SourceState), source registration,
//!   POCQ event loop.

mod graph;
mod handle;
mod node;
mod runner;

pub use handle::{Handle, InputKindsHandles};

use std::any::TypeId;

use crate::operator::Operator;
use crate::store::Store;
use crate::types::{InputKinds, Scalar};

use graph::Graph;
use node::{new_closure, new_node};
use runner::SourceState;

/// Store-based DAG runtime.
///
/// Wraps a [`Graph`] and provides a type-safe API for node creation, operator
/// registration, store access, and event-driven execution.
///
/// # Type-safe API example
///
/// ```ignore
/// let mut sc = Scenario::new();
///
/// // Create nodes (non-windowed by default).
/// let ha = sc.create_node::<f64>(&[], &[0.0]);
/// let hb = sc.create_node::<f64>(&[], &[0.0]);
///
/// // Register an operator.  Inputs are validated via TypeId at registration.
/// // The operator's window_sizes() auto-promotes input stores if needed.
/// let hc = sc.add_operator([ha, hb], my_add_op);
///
/// // Write to source nodes and flush.
/// sc.store_mut(ha).push(1, &[10.0]);
/// sc.store_mut(hb).push(1, &[3.0]);
/// sc.flush(1, &[ha.index(), hb.index()]);
///
/// assert_eq!(sc.store(hc).current(), &[13.0]);
/// ```
pub struct Scenario {
    graph: Graph,
    source_states: Vec<SourceState>,
}

impl Scenario {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            source_states: Vec::new(),
        }
    }

    // -- Node creation -------------------------------------------------------

    /// Create a node.  Starts as `window = 1` (single element, no history).
    /// Operators auto-promote inputs via `window_sizes()`, or call
    /// `store_mut(h).ensure_min_window(n)` manually.
    pub fn create_node<T: Scalar>(&mut self, shape: &[usize], default: &[T]) -> Handle<T> {
        let store = Store::element(shape, default);
        let idx = self.graph.push_node(new_node(store));
        Handle::new(idx)
    }

    // -- Store access --------------------------------------------------------

    /// Immutable access to a node's store.  Panics on TypeId mismatch.
    #[inline(always)]
    pub fn store<T: Scalar>(&self, h: Handle<T>) -> &Store<T> {
        let node = &self.graph.nodes[h.index()];
        assert_eq!(
            node.type_id,
            TypeId::of::<T>(),
            "type mismatch at node {}",
            h.index(),
        );
        unsafe { &*(node.store as *const Store<T>) }
    }

    /// Mutable access to a node's store.  Panics on TypeId mismatch.
    #[inline(always)]
    pub fn store_mut<T: Scalar>(&mut self, h: Handle<T>) -> &mut Store<T> {
        let node = &self.graph.nodes[h.index()];
        assert_eq!(
            node.type_id,
            TypeId::of::<T>(),
            "type mismatch at node {}",
            h.index(),
        );
        unsafe { &mut *(node.store as *mut Store<T>) }
    }

    // -- Operator registration -----------------------------------------------

    /// Register an operator, creating its output node.
    ///
    /// 1. Validates input handles via TypeId.
    /// 2. Auto-promotes input stores per `op.window_sizes()`.
    /// 3. Collects input store pointers for the closure.
    /// 4. Creates the output node (filled with `op.output()` default value)
    ///    and wires edges.
    /// 5. Attaches the closure.
    pub fn add_operator<Op>(
        &mut self,
        inputs: impl Into<<Op::Inputs as InputKindsHandles>::Handles>,
        op: Op,
    ) -> Handle<Op::Scalar>
    where
        Op: Operator,
        Op::Inputs: InputKindsHandles,
    {
        let handles = inputs.into();
        let node_ids = <Op::Inputs as InputKindsHandles>::node_ids(&handles);

        // 1. Validate TypeIds, collect input shapes.
        for &(idx, expected_tid) in node_ids.iter() {
            assert!(
                idx < self.graph.len(),
                "invalid handle: node index {idx} out of range",
            );
            assert_eq!(
                self.graph.nodes[idx].type_id, expected_tid,
                "type mismatch at node {idx}",
            );
        }
        let input_shapes: Box<[&[usize]]> = node_ids
            .iter()
            .map(|&(idx, _)| self.graph.nodes[idx].shape())
            .collect();

        // 2. Auto-promote input stores per operator's window_sizes.
        let window_sizes = op.window_sizes(&input_shapes);
        let store_ptrs: Vec<*mut u8> = node_ids
            .iter()
            .map(|&(idx, _)| self.graph.nodes[idx].store)
            .collect();
        unsafe { <Op::Inputs as InputKinds>::promote(&store_ptrs, &window_sizes) };

        // 3. Collect input store pointers (as *const for the closure).
        let input_ptrs: Box<[*const u8]> = store_ptrs.iter().map(|&p| p as *const u8).collect();

        // 4. Create output node.
        let (output_shape, default) = op.default(&input_shapes);
        let state = op.init();

        let output_store = Store::element(&output_shape, &default);
        let output_idx = self.graph.push_node(new_node(output_store));

        // Wire edges: each input node -> output node.
        for &(input_idx, _) in node_ids.iter() {
            self.graph.add_edge(input_idx, output_idx);
        }

        // 5. Attach closure.
        let closure = new_closure::<Op>(input_ptrs, state);
        self.graph.nodes[output_idx].closure = Some(closure);

        Handle::new(output_idx)
    }

    // -- Flush ---------------------------------------------------------------

    /// Propagate updates through the DAG.
    pub fn flush(&mut self, timestamp: i64, updated_sources: &[usize]) {
        self.graph.flush(timestamp, updated_sources);
    }

    // -- Low-level accessors (for bridge / FFI) ------------------------------

    /// Raw store pointer for a node.  Used by the bridge to create views.
    ///
    /// # Safety
    ///
    /// The caller must know the scalar type `T` and cast accordingly.
    #[cfg(feature = "python")]
    pub(crate) fn node_store_ptr(&self, index: usize) -> *mut u8 {
        self.graph.nodes[index].store
    }

    /// Element stride (scalars per element) for a node.
    #[cfg(feature = "python")]
    pub(crate) fn node_stride(&self, index: usize) -> usize {
        self.graph.nodes[index].stride()
    }

    /// Element shape for a node.
    #[cfg(feature = "python")]
    pub(crate) fn node_shape(&self, index: usize) -> &[usize] {
        self.graph.nodes[index].shape()
    }

    /// Number of nodes in the graph.
    #[cfg(feature = "python")]
    #[allow(dead_code)]
    pub(crate) fn node_count(&self) -> usize {
        self.graph.len()
    }

    /// Add a directed edge between two nodes.
    #[cfg(feature = "python")]
    pub(crate) fn add_edge(&mut self, from: usize, to: usize) {
        self.graph.add_edge(from, to);
    }

    /// Attach a raw type-erased closure to a node.
    ///
    /// This is used by the bridge to attach Python operator callbacks.
    /// The `compute_fn` has the same signature as `ComputeFn` in the node
    /// module: `unsafe fn(&[*const u8], *mut u8, *mut u8, i64) -> bool`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// * `input_ptrs` point to valid stores for the lifetime of the node.
    /// * `state` is a valid heap-allocated state object.
    /// * `compute_fn` correctly interprets the pointers.
    #[cfg(feature = "python")]
    pub(crate) fn attach_raw_closure(
        &mut self,
        node_index: usize,
        input_ptrs: Box<[*const u8]>,
        compute_fn: unsafe fn(&[*const u8], *mut u8, *mut u8, i64) -> bool,
        state: Box<dyn std::any::Any + Send>,
    ) {
        use node::Closure;

        let state_ptr = Box::into_raw(state) as *mut u8;

        /// Drop a heap-allocated `Box<dyn Any + Send>`.
        ///
        /// # Safety
        ///
        /// `ptr` must have been created by `Box::into_raw(Box::new(..))`.
        unsafe fn drop_dyn_state(ptr: *mut u8) {
            unsafe { drop(Box::from_raw(ptr as *mut Box<dyn std::any::Any + Send>)) };
        }

        self.graph.nodes[node_index].closure = Some(Closure {
            compute_fn,
            state: state_ptr,
            input_ptrs,
            drop_state: drop_dyn_state,
        });
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
    use crate::store::{ElementViewMut, Store};

    /// Minimal binary add operator for testing.
    struct TestAdd;

    impl Operator for TestAdd {
        type State = ();
        type Inputs = (Store<f64>, Store<f64>);
        type Scalar = f64;

        fn window_sizes(&self, _: &[&[usize]]) -> (usize, usize) {
            (1, 1)
        }

        fn default(&self, input_shapes: &[&[usize]]) -> (Box<[usize]>, Box<[f64]>) {
            let shape: Box<[usize]> = input_shapes[0].into();
            let stride = shape.iter().product::<usize>();
            (shape, vec![0.0; stride].into())
        }

        fn init(self) {}

        fn compute(
            _state: &mut (),
            inputs: (&Store<f64>, &Store<f64>),
            output: ElementViewMut<'_, f64>,
        ) -> bool {
            let (a, b) = inputs;
            let (a, b) = (a.current(), b.current());
            for i in 0..output.values.len() {
                output.values[i] = a[i] + b[i];
            }
            true
        }
    }

    #[test]
    fn scenario_simple_add() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let hc = sc.add_operator([ha, hb], TestAdd);

        // Write new values and flush.
        sc.store_mut(ha).push(1, &[10.0]);
        sc.store_mut(hb).push(1, &[3.0]);
        sc.flush(1, &[ha.index(), hb.index()]);

        assert_eq!(sc.store(hc).current(), &[13.0]);
    }

    #[test]
    fn scenario_windowed_output() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let hc = sc.add_operator([ha, hb], TestAdd);

        // Promote output to unbounded so it keeps full history.
        sc.store_mut(hc).ensure_min_window(0);

        sc.store_mut(ha).push(1, &[10.0]);
        sc.store_mut(hb).push(1, &[3.0]);
        sc.flush(1, &[ha.index(), hb.index()]);

        sc.store_mut(ha).push(2, &[20.0]);
        sc.store_mut(hb).push(2, &[7.0]);
        sc.flush(2, &[ha.index(), hb.index()]);

        let store = sc.store(hc);
        assert_eq!(store.len(), 3); // initial + 2 flushes
        assert_eq!(store.timestamps(), &[i64::MIN, 1, 2]);
        assert_eq!(store.values(), &[0.0, 13.0, 27.0]);
    }

    #[test]
    fn scenario_chain() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let hab = sc.add_operator([ha, hb], TestAdd);

        /// Minimal multiply operator.
        struct TestMul;

        impl Operator for TestMul {
            type State = ();
            type Inputs = (Store<f64>, Store<f64>);
            type Scalar = f64;

            fn window_sizes(&self, _: &[&[usize]]) -> (usize, usize) {
                (1, 1)
            }

            fn default(&self, s: &[&[usize]]) -> (Box<[usize]>, Box<[f64]>) {
                let shape: Box<[usize]> = s[0].into();
                let stride = shape.iter().product::<usize>();
                (shape, vec![0.0; stride].into())
            }

            fn init(self) {}

            fn compute(
                _: &mut (),
                inputs: (&Store<f64>, &Store<f64>),
                output: ElementViewMut<'_, f64>,
            ) -> bool {
                let (a, b) = (inputs.0.current(), inputs.1.current());
                for i in 0..output.values.len() {
                    output.values[i] = a[i] * b[i];
                }
                true
            }
        }

        let hout = sc.add_operator([hab, ha], TestMul);

        sc.store_mut(ha).push(1, &[2.0]);
        sc.store_mut(hb).push(1, &[3.0]);
        sc.flush(1, &[ha.index(), hb.index()]);

        // (2+3) * 2 = 10
        assert_eq!(sc.store(hout).current(), &[10.0]);
    }

    // -- POCQ run tests (async, using real operators + ArraySource) ----------

    #[tokio::test]
    async fn scenario_run_single_source() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source(
            ArraySource::new(vec![1, 2, 3], vec![10.0, 20.0, 30.0], 1),
            true,
        );

        sc.run().await;

        let store = sc.store(ha);
        assert_eq!(store.len(), 4); // initial + 3
        assert_eq!(store.timestamps(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(store.values(), &[0.0, 10.0, 20.0, 30.0]);
    }

    #[tokio::test]
    async fn scenario_run_two_sources_interleaved() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 3], vec![10.0, 30.0], 1), false);
        let hb = sc.add_source(ArraySource::new(vec![2, 3], vec![20.0, 40.0], 1), false);
        let ho = sc.add_operator([ha, hb], TestAdd);
        sc.store_mut(ho).ensure_min_window(0); // keep history

        sc.run().await;

        let store = sc.store(ho);
        // initial(0), ts=1: 10+0=10, ts=2: 10+20=30, ts=3: 30+40=70
        assert_eq!(store.len(), 4);
        assert_eq!(store.timestamps(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(store.values(), &[0.0, 10.0, 30.0, 70.0]);
    }

    #[tokio::test]
    async fn scenario_run_coalescing() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2], vec![10.0, 20.0], 1), true);
        let hb = sc.add_source(ArraySource::new(vec![1, 2], vec![100.0, 200.0], 1), false);
        let ho = sc.add_operator([ha, hb], TestAdd);
        sc.store_mut(ho).ensure_min_window(0);

        sc.run().await;

        let out = sc.store(ho);
        assert_eq!(out.len(), 3); // initial + 2 coalesced
        assert_eq!(out.timestamps(), &[i64::MIN, 1, 2]);
        assert_eq!(out.values(), &[0.0, 110.0, 220.0]);

        let a = sc.store(ha);
        assert_eq!(a.len(), 3); // initial + 2
    }

    #[tokio::test]
    async fn scenario_run_chained() {
        use crate::sources::ArraySource;

        struct TestMul2;

        impl crate::operator::Operator for TestMul2 {
            type State = ();
            type Inputs = (Store<f64>, Store<f64>);
            type Scalar = f64;

            fn window_sizes(&self, _: &[&[usize]]) -> (usize, usize) {
                (1, 1)
            }

            fn default(&self, s: &[&[usize]]) -> (Box<[usize]>, Box<[f64]>) {
                let shape: Box<[usize]> = s[0].into();
                let stride = shape.iter().product::<usize>();
                (shape, vec![0.0; stride].into())
            }

            fn init(self) {}

            fn compute(
                _: &mut (),
                inputs: (&Store<f64>, &Store<f64>),
                output: ElementViewMut<'_, f64>,
            ) -> bool {
                let (a, b) = (inputs.0.current(), inputs.1.current());
                for i in 0..output.values.len() {
                    output.values[i] = a[i] * b[i];
                }
                true
            }
        }

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2], vec![2.0, 5.0], 1), false);
        let hb = sc.add_source(ArraySource::new(vec![1, 2], vec![3.0, 10.0], 1), false);
        let hab = sc.add_operator([ha, hb], TestAdd);
        let hout = sc.add_operator([hab, ha], TestMul2);
        sc.store_mut(hout).ensure_min_window(0);

        sc.run().await;

        let out = sc.store(hout);
        assert_eq!(out.len(), 3);
        // initial: 0, ts=1: (2+3)*2=10, ts=2: (5+10)*5=75
        assert_eq!(out.values(), &[0.0, 10.0, 75.0]);
    }

    #[tokio::test]
    async fn scenario_run_filter() {
        use crate::operators::Filter;
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source(
            ArraySource::new(vec![1, 2, 3, 4], vec![1.0, 5.0, 2.0, 10.0], 1),
            false,
        );
        let ho = sc.add_operator([ha], Filter::new(|v: &[f64]| v[0] > 3.0));
        sc.store_mut(ho).ensure_min_window(0);

        sc.run().await;

        let out = sc.store(ho);
        // initial(0) + passes: ts=2(5.0), ts=4(10.0)
        assert_eq!(out.len(), 3);
        assert_eq!(out.timestamps(), &[i64::MIN, 2, 4]);
        assert_eq!(out.values(), &[0.0, 5.0, 10.0]);
    }
}
