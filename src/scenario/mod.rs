//! Scenario — the DAG runtime for event-driven computation.
//!
//! A [`Scenario`] is a directed acyclic graph of nodes, where each node holds
//! an arbitrary value of type `T: Send + 'static`.  Nodes are fed by
//! [`Source`](crate::source::Source)s and connected by
//! [`Operator`](crate::operator::Operator)s.
//!
//! # Architecture
//!
//! Internally, nodes are stored as type-erased `(pointer, TypeId)` slots.
//! Type safety is enforced at registration time via [`Handle<T>`] and
//! [`TypeId`] checks.  After registration, operator dispatch uses raw pointer
//! casts through monomorphised function pointers — zero dynamic dispatch
//! overhead on the hot path.

mod graph;
pub mod handle;
mod node;
mod runner;

pub use handle::{Handle, InputKindsHandles};

use std::any::TypeId;

use crate::operator::Operator;
use crate::source::Source;
use crate::types::InputKinds;

use graph::Graph;
use node::{Closure, Node};
use runner::SourceState;

/// Type-erased DAG runtime for event-driven computation.
///
/// # Type-safe API example
///
/// ```ignore
/// use tradingflow::array::Array;
///
/// let mut sc = Scenario::new();
///
/// let ha = sc.create_node(Array::scalar(0.0_f64));
/// let hb = sc.create_node(Array::scalar(0.0_f64));
/// let hc = sc.add_operator(my_add_op, (ha, hb));
///
/// sc.value_mut(ha)[0] = 10.0;
/// sc.value_mut(hb)[0] = 3.0;
/// sc.flush(1, &[ha.index(), hb.index()]);
///
/// assert_eq!(sc.value(hc).as_slice(), &[13.0]);
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

    // -- Value access --------------------------------------------------------

    /// Immutable access to a node's value.  Panics on TypeId mismatch.
    #[inline(always)]
    pub fn value<T: Send + 'static>(&self, h: Handle<T>) -> &T {
        let node = &self.graph.nodes[h.index()];
        assert_eq!(
            node.type_id,
            TypeId::of::<T>(),
            "type mismatch at node {}",
            h.index(),
        );
        unsafe { &*(node.value as *const T) }
    }

    /// Mutable access to a node's value.  Panics on TypeId mismatch.
    #[inline(always)]
    pub fn value_mut<T: Send + 'static>(&mut self, h: Handle<T>) -> &mut T {
        let node = &self.graph.nodes[h.index()];
        assert_eq!(
            node.type_id,
            TypeId::of::<T>(),
            "type mismatch at node {}",
            h.index(),
        );
        unsafe { &mut *(node.value as *mut T) }
    }

    // -- Node creation -------------------------------------------------------

    /// Create a bare node with an initial value.
    pub fn create_node<T: Send + 'static>(&mut self, value: T) -> Handle<T> {
        let idx = self.graph.add_node(Node::new(value));
        Handle::new(idx)
    }

    /// Register a [`Source`], creating the output node.
    ///
    /// Sources that use [`tokio::spawn`] internally (e.g. [`ArraySource`],
    /// [`IterSource`]) require a tokio runtime to be active.
    pub fn add_source<S: Source>(&mut self, source: S) -> Handle<S::Output> {
        let (hist_rx, live_rx, output) = source.init(i64::MIN);
        let idx = self.graph.add_node(Node::new(output));
        self.source_states
            .push(SourceState::new::<S>(idx, hist_rx, live_rx));
        Handle::new(idx)
    }

    /// Register an [`Operator`], creating its output node.
    ///
    /// All inputs are trigger edges — the operator fires whenever any
    /// input updates.  Use [`add_operator_periodic`](Scenario::add_operator_periodic)
    /// to trigger on a clock instead.
    pub fn add_operator<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputKindsHandles>::Handles>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputKindsHandles,
    {
        self.register_operator_from_handles(operator, inputs, None)
    }

    /// Register an [`Operator`] that fires on a clock instead of on
    /// input updates.
    ///
    /// The operator reads from `inputs` but is only scheduled when the
    /// `clock` node updates.  The clock is not an input — the operator
    /// does not read its value.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let prices = sc.add_source(daily_prices);
    /// let clock = sc.add_source(monthly_clock);
    ///
    /// // Fires monthly, reads daily prices.
    /// let h = sc.add_operator_periodic(my_op, (prices,), clock);
    /// ```
    pub fn add_operator_periodic<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputKindsHandles>::Handles>,
        clock: Handle<()>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputKindsHandles,
    {
        self.register_operator_from_handles(operator, inputs, Some(clock))
    }

    // -- Untyped registration (for FFI / bridge) ------------------------------

    /// Register a [`Source`], returning the output node index instead of a
    /// typed [`Handle`].
    pub fn add_source_untyped<S: Source>(&mut self, source: S) -> usize {
        self.add_source(source).index()
    }

    /// `TypeId` of a node's value.
    pub fn node_type_id(&self, index: usize) -> TypeId {
        self.graph.nodes[index].type_id
    }

    // -- Typed registration --------------------------------------------------

    /// Register an operator from typed handles.
    fn register_operator_from_handles<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputKindsHandles>::Handles>,
        clock: Option<Handle<()>>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputKindsHandles,
    {
        let input_indices = <O::Inputs as InputKindsHandles>::node_indices(&inputs.into());
        let clock_idx = clock.map(|c| c.index());
        let output_idx =
            self.register_operator_from_indices::<O>(operator, &input_indices, clock_idx);
        Handle::new(output_idx)
    }

    /// Register an operator from raw node indices.
    ///
    /// Validates `TypeId`s, collects input pointers, calls `init`, creates
    /// the output node, wires trigger edges, and attaches the closure.
    ///
    /// If `clock` is `None`, all inputs are trigger edges.
    /// If `clock` is `Some(idx)`, only the clock node triggers.
    pub fn register_operator_from_indices<O: Operator>(
        &mut self,
        operator: O,
        input_indices: &[usize],
        clock_index: Option<usize>,
    ) -> usize {
        // Validate arity.
        let expected_tids = <O::Inputs as InputKinds>::type_ids(input_indices.len());
        assert_eq!(
            input_indices.len(),
            expected_tids.len(),
            "arity mismatch: operator expects {} inputs, got {}",
            expected_tids.len(),
            input_indices.len(),
        );

        // 1. Validate TypeIds.
        for (i, (&idx, &expected_tid)) in input_indices.iter().zip(expected_tids.iter()).enumerate()
        {
            assert!(
                idx < self.graph.len(),
                "invalid index: node {idx} out of range",
            );
            assert_eq!(
                self.graph.nodes[idx].type_id, expected_tid,
                "type mismatch at input {i} (node {idx})",
            );
        }

        // 2. Collect input value pointers.
        let input_ptrs: Box<[*const u8]> = input_indices
            .iter()
            .map(|&idx| self.graph.nodes[idx].value as *const u8)
            .collect();

        // 3. Call init eagerly.
        let input_refs = unsafe { <O::Inputs as InputKinds>::from_ptrs(&input_ptrs) };
        let (state, output) = operator.init(input_refs, i64::MIN);

        // 4. Create output node.
        let output_idx = self.graph.add_node(Node::new(output));

        // 5. Wire trigger edges.
        match clock_index {
            None => {
                for &input_idx in input_indices {
                    self.graph.add_trigger_edge(input_idx, output_idx);
                }
            }
            Some(clock_idx) => {
                self.graph.add_trigger_edge(clock_idx, output_idx);
            }
        }

        // 6. Attach closure.
        let closure = Closure::from_state::<O>(state, input_ptrs);
        self.graph.nodes[output_idx].closure = Some(closure);

        output_idx
    }

    // -- Flush ---------------------------------------------------------------

    /// Propagate updates through the DAG.
    pub fn flush(&mut self, timestamp: i64, updated_sources: &[usize]) {
        self.graph.flush(timestamp, updated_sources);
    }

    // -- Low-level accessors (for bridge / FFI) ------------------------------

    /// Raw value pointer for a node.
    #[cfg(feature = "python")]
    pub(crate) fn node_value_ptr(&self, index: usize) -> *mut u8 {
        self.graph.nodes[index].value
    }

    /// Add a directed edge between two nodes.
    #[cfg(feature = "python")]
    pub(crate) fn add_trigger_edge(&mut self, from: usize, to: usize) {
        self.graph.add_trigger_edge(from, to);
    }

    /// Attach a raw type-erased closure to a node.
    ///
    /// Used by the bridge to attach Python operator callbacks.
    /// Attach a raw type-erased closure to a node.
    ///
    /// The caller provides a concrete state type `S`, which is heap-allocated
    /// by this method.  The `compute_fn` must cast `state_ptr` to `*mut S`.
    ///
    /// # Safety
    ///
    /// * `input_ptrs` must point to valid node values for the node's lifetime.
    /// * `compute_fn` must correctly interpret the pointer types.
    #[cfg(feature = "python")]
    pub(crate) fn attach_raw_closure<S: Send + 'static>(
        &mut self,
        node_index: usize,
        input_ptrs: Box<[*const u8]>,
        compute_fn: unsafe fn(&[*const u8], *mut u8, *mut u8, i64) -> bool,
        state: Box<S>,
    ) {
        let state_ptr = Box::into_raw(state) as *mut u8;

        /// # Safety
        /// `ptr` must have been created by `Box::into_raw(Box::new(..))` for type `T`.
        unsafe fn drop_state<T>(ptr: *mut u8) {
            unsafe { drop(Box::from_raw(ptr as *mut T)) };
        }

        self.graph.nodes[node_index].closure = Some(Closure::new(
            compute_fn,
            input_ptrs,
            state_ptr,
            drop_state::<S>,
        ));
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
    use crate::array::Array;
    use crate::operators::{Filter, Record, add};
    use crate::series::Series;
    use crate::sources::ArraySource;

    #[test]
    fn scenario_simple_add() {
        let mut sc = Scenario::new();
        let ha = sc.create_node(Array::scalar(0.0_f64));
        let hb = sc.create_node(Array::scalar(0.0_f64));
        let hc = sc.add_operator(add::<f64>(), (ha, hb));

        sc.value_mut(ha)[0] = 10.0;
        sc.value_mut(hb)[0] = 3.0;
        sc.flush(1, &[ha.index(), hb.index()]);

        assert_eq!(sc.value(hc).as_slice(), &[13.0]);
    }

    #[test]
    fn scenario_strided_add() {
        let mut sc = Scenario::new();
        let ha = sc.create_node(Array::from_vec(&[2], vec![1.0_f64, 2.0]));
        let hb = sc.create_node(Array::from_vec(&[2], vec![10.0_f64, 20.0]));
        let hc = sc.add_operator(add::<f64>(), (ha, hb));

        sc.flush(1, &[ha.index(), hb.index()]);
        assert_eq!(sc.value(hc).as_slice(), &[11.0, 22.0]);
    }

    #[test]
    fn scenario_chain() {
        let mut sc = Scenario::new();
        let ha = sc.create_node(Array::scalar(2.0_f64));
        let hb = sc.create_node(Array::scalar(3.0_f64));
        let hab = sc.add_operator(add::<f64>(), (ha, hb));

        use crate::operators::multiply;
        let hout = sc.add_operator(multiply::<f64>(), (hab, ha));

        sc.flush(1, &[ha.index(), hb.index()]);
        // (2+3) * 2 = 10
        assert_eq!(sc.value(hout).as_slice(), &[10.0]);
    }

    #[test]
    fn scenario_record() {
        let mut sc = Scenario::new();
        let ha = sc.create_node(Array::scalar(0.0_f64));
        let hb = sc.create_node(Array::scalar(0.0_f64));
        let hsum = sc.add_operator(add::<f64>(), (ha, hb));
        let hseries = sc.add_operator(Record::<f64>::new(), (hsum,));

        sc.value_mut(ha)[0] = 10.0;
        sc.value_mut(hb)[0] = 3.0;
        sc.flush(1, &[ha.index(), hb.index()]);

        sc.value_mut(ha)[0] = 20.0;
        sc.value_mut(hb)[0] = 7.0;
        sc.flush(2, &[ha.index(), hb.index()]);

        let series: &Series<f64> = sc.value(hseries);
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[1, 2]);
        assert_eq!(series.values(), &[13.0, 27.0]);
    }

    // -- POCQ run tests (async) -----------------------------------------------

    #[tokio::test]
    async fn scenario_run_single_source() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2, 3], vec![10.0, 20.0, 30.0], 1));
        let hseries = sc.add_operator(Record::<f64>::new(), (ha,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hseries);
        assert_eq!(series.len(), 3);
        assert_eq!(series.timestamps(), &[1, 2, 3]);
        assert_eq!(series.values(), &[10.0, 20.0, 30.0]);
    }

    #[tokio::test]
    async fn scenario_run_two_sources_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 3], vec![10.0, 30.0], 1));
        let hb = sc.add_source(ArraySource::new(vec![2, 3], vec![20.0, 40.0], 1));
        let ho = sc.add_operator(add::<f64>(), (ha, hb));
        let hseries = sc.add_operator(Record::<f64>::new(), (ho,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hseries);
        // ts=1: 10+0=10, ts=2: 10+20=30, ts=3: 30+40=70
        assert_eq!(series.len(), 3);
        assert_eq!(series.timestamps(), &[1, 2, 3]);
        assert_eq!(series.values(), &[10.0, 30.0, 70.0]);
    }

    #[tokio::test]
    async fn scenario_run_coalescing() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2], vec![10.0, 20.0], 1));
        let hb = sc.add_source(ArraySource::new(vec![1, 2], vec![100.0, 200.0], 1));
        let ho = sc.add_operator(add::<f64>(), (ha, hb));
        let hseries = sc.add_operator(Record::<f64>::new(), (ho,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hseries);
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[1, 2]);
        assert_eq!(series.values(), &[110.0, 220.0]);
    }

    #[tokio::test]
    async fn scenario_run_chained() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2], vec![2.0, 5.0], 1));
        let hb = sc.add_source(ArraySource::new(vec![1, 2], vec![3.0, 10.0], 1));
        let hab = sc.add_operator(add::<f64>(), (ha, hb));

        use crate::operators::multiply;
        let hout = sc.add_operator(multiply::<f64>(), (hab, ha));
        let hseries = sc.add_operator(Record::<f64>::new(), (hout,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hseries);
        assert_eq!(series.len(), 2);
        // ts=1: (2+3)*2=10, ts=2: (5+10)*5=75
        assert_eq!(series.values(), &[10.0, 75.0]);
    }

    #[tokio::test]
    async fn scenario_run_filter() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            vec![1, 2, 3, 4],
            vec![1.0, 5.0, 2.0, 10.0],
            1,
        ));
        let ho = sc.add_operator(Filter::new(|v: &Array<f64>| v[0] > 3.0), (ha,));
        let hseries = sc.add_operator(Record::<f64>::new(), (ho,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hseries);
        // passes: ts=2(5.0), ts=4(10.0)
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[2, 4]);
        assert_eq!(series.values(), &[5.0, 10.0]);
    }

    #[test]
    fn scenario_register_operator_from_indices() {
        let mut sc = Scenario::new();
        let ha = sc.create_node(Array::scalar(0.0_f64));
        let hb = sc.create_node(Array::scalar(0.0_f64));
        let out_idx =
            sc.register_operator_from_indices(add::<f64>(), &[ha.index(), hb.index()], None);

        sc.value_mut(ha)[0] = 10.0;
        sc.value_mut(hb)[0] = 3.0;
        sc.flush(1, &[ha.index(), hb.index()]);

        let out = unsafe { &*(sc.graph.nodes[out_idx].value as *const Array<f64>) };
        assert_eq!(out.as_slice(), &[13.0]);
    }

    #[test]
    #[should_panic(expected = "type mismatch")]
    fn scenario_register_operator_from_indices_type_mismatch() {
        let mut sc = Scenario::new();
        let ha = sc.create_node(Array::scalar(0.0_f64));
        let hb = sc.create_node(Array::scalar(0_i32));
        // add::<f64> expects (Array<f64>, Array<f64>) but hb is Array<i32>
        sc.register_operator_from_indices(add::<f64>(), &[ha.index(), hb.index()], None);
    }

    #[test]
    fn scenario_arbitrary_type() {
        use std::collections::BTreeMap;

        let mut sc = Scenario::new();
        let h = sc.create_node(BTreeMap::<String, f64>::new());

        sc.value_mut(h).insert("price".to_string(), 42.0);
        assert_eq!(sc.value(h).get("price"), Some(&42.0));
    }

    // -- Periodic operator tests -----------------------------------------------

    #[tokio::test]
    async fn scenario_periodic_single_input() {
        // Source A updates at ts 1,2,3.  Clock fires at ts 2.
        // Operator reads A, triggered only by clock.
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2, 3], vec![10.0, 20.0, 30.0], 1));
        let hclock = sc.add_source(clock(vec![2]));

        let ho = sc.add_operator_periodic(Filter::new(|_: &Array<f64>| true), (ha,), hclock);
        let hs = sc.add_operator(Record::<f64>::new(), (ho,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hs);
        // Only ts=2 triggers: A=20
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), &[2]);
        assert_eq!(series.values(), &[20.0]);
    }

    #[tokio::test]
    async fn scenario_periodic_two_inputs() {
        // A updates at 1,2,3; B updates at 1,3. Clock fires at 2.
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0], 1));
        let hb = sc.add_source(ArraySource::new(vec![1, 3], vec![10.0, 30.0], 1));
        let hclock = sc.add_source(clock(vec![2]));

        let ho = sc.add_operator_periodic(add::<f64>(), (ha, hb), hclock);
        let hs = sc.add_operator(Record::<f64>::new(), (ho,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hs);
        // Clock fires at ts=2. At ts=2: A=2, B=10 (carry from ts=1).
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), &[2]);
        assert_eq!(series.values(), &[12.0]);
    }

    #[tokio::test]
    async fn scenario_periodic_multiple_ticks() {
        // Daily prices, monthly clock at ts=2 and ts=4.
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            vec![1, 2, 3, 4, 5],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            1,
        ));
        let hclock = sc.add_source(clock(vec![2, 4]));

        let ho = sc.add_operator_periodic(Filter::new(|_: &Array<f64>| true), (ha,), hclock);
        let hs = sc.add_operator(Record::<f64>::new(), (ho,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hs);
        // ts=2: A=20, ts=4: A=40
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[2, 4]);
        assert_eq!(series.values(), &[20.0, 40.0]);
    }
}
