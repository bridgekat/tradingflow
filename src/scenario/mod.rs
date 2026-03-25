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
//!
//! # Operator registration
//!
//! All operator registration flows through [`Scenario::add_erased_operator`],
//! which accepts an [`ErasedOperator`].  Typed convenience methods
//! ([`add_operator`](Scenario::add_operator),
//! [`register_operator_from_indices`](Scenario::register_operator_from_indices))
//! construct an [`ErasedOperator`] from a concrete [`Operator`] and delegate.

mod graph;
pub mod handle;
mod node;
mod runner;

pub use handle::{Handle, InputTypesHandles};
pub use node::{ComputeFn, ErasedOperator, drop_fn};

use std::any::TypeId;

use crate::operator::Operator;
use crate::source::Source;
use crate::types::InputTypes;

use graph::Graph;
use node::{Closure, Node};
use runner::SourceState;

/// Type-erased DAG runtime for event-driven computation.
///
/// # Type-safe API example
///
/// ```
/// use tradingflow::{Scenario, Array};
/// use tradingflow::operators::add;
///
/// let mut sc = Scenario::new();
///
/// let ha = sc.create_node(Array::scalar(0.0));
/// let hb = sc.create_node(Array::scalar(0.0));
/// let hc = sc.add_operator(add::<f64>(), (ha, hb));
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

    // -- Typed operator registration -----------------------------------------

    /// Register an [`Operator`], creating its output node.
    ///
    /// All inputs are trigger edges — the operator fires whenever any
    /// input updates.  Use [`add_operator_periodic`](Scenario::add_operator_periodic)
    /// to trigger on a clock instead.
    pub fn add_operator<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputTypesHandles>::Handles>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputTypesHandles,
    {
        self.register_operator_from_handles(operator, inputs, None)
    }

    /// Register an [`Operator`] that fires on a clock instead of on
    /// input updates.
    pub fn add_operator_periodic<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputTypesHandles>::Handles>,
        clock: Handle<()>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputTypesHandles,
    {
        self.register_operator_from_handles(operator, inputs, Some(clock))
    }

    /// Register an operator from typed handles.
    fn register_operator_from_handles<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputTypesHandles>::Handles>,
        clock: Option<Handle<()>>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputTypesHandles,
    {
        let input_indices = <O::Inputs as InputTypesHandles>::node_indices(&inputs.into());
        let clock_idx = clock.map(|c| c.index());
        let output_idx =
            self.register_operator_from_indices::<O>(operator, &input_indices, clock_idx);
        Handle::new(output_idx)
    }

    /// Register an operator from raw node indices.
    ///
    /// Validates `TypeId`s, calls [`Operator::init`] via
    /// [`ErasedOperator::from_operator`], then delegates to
    /// [`add_erased_operator`](Scenario::add_erased_operator).
    pub fn register_operator_from_indices<O: Operator>(
        &mut self,
        operator: O,
        input_indices: &[usize],
        clock_index: Option<usize>,
    ) -> usize {
        // Validate arity + TypeIds before calling init.
        let expected_tids = <O::Inputs as InputTypes>::type_ids(input_indices.len());
        assert_eq!(
            input_indices.len(),
            expected_tids.len(),
            "arity mismatch: operator expects {} inputs, got {}",
            expected_tids.len(),
            input_indices.len(),
        );
        for (i, (&idx, &expected_tid)) in
            input_indices.iter().zip(expected_tids.iter()).enumerate()
        {
            assert!(idx < self.graph.len(), "invalid index: node {idx} out of range");
            assert_eq!(
                self.graph.nodes[idx].type_id, expected_tid,
                "type mismatch at input {i} (node {idx})",
            );
        }

        // Collect validated input pointers and call init.
        let input_ptrs: Box<[*const u8]> = input_indices
            .iter()
            .map(|&idx| self.graph.nodes[idx].value as *const u8)
            .collect();
        // SAFETY: input pointers are valid — we just validated TypeIds above.
        let erased = unsafe { ErasedOperator::from_operator(operator, &input_ptrs, i64::MIN) };
        self.add_erased_operator(erased, input_indices, clock_index)
    }

    // -- Erased operator registration ----------------------------------------

    /// Register a pre-initialized, type-erased operator.
    ///
    /// This is the unified registration method used by both the typed Rust
    /// API and the Python bridge.
    ///
    /// 1. Validates input `TypeId`s against `erased.input_type_ids`.
    /// 2. Transfers the pre-initialized output into a new node.
    /// 3. Wires trigger edges (inputs or clock).
    /// 4. Attaches the compute closure with pre-initialized state.
    ///
    /// Returns the output node index.
    pub fn add_erased_operator(
        &mut self,
        mut erased: ErasedOperator,
        input_indices: &[usize],
        clock_index: Option<usize>,
    ) -> usize {
        // 1. Validate arity.
        assert_eq!(
            input_indices.len(),
            erased.input_type_ids.len(),
            "arity mismatch: operator expects {} inputs, got {}",
            erased.input_type_ids.len(),
            input_indices.len(),
        );

        // 2. Validate TypeIds.
        for (i, (&idx, &expected_tid)) in
            input_indices.iter().zip(erased.input_type_ids.iter()).enumerate()
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

        // 3. Collect input value pointers for the closure.
        let input_ptrs: Box<[*const u8]> = input_indices
            .iter()
            .map(|&idx| self.graph.nodes[idx].value as *const u8)
            .collect();

        // 4. Take ownership of pre-initialized state + output.
        let (state_ptr, output_ptr) = erased.take_ptrs();

        // 5. Create output node.
        let output_idx = self.graph.add_node(Node::from_raw(
            erased.output_type_id,
            output_ptr,
            erased.output_drop_fn,
        ));

        // 6. Wire trigger edges.
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

        // 7. Attach closure.
        self.graph.nodes[output_idx].closure = Some(Closure::new(
            erased.compute_fn,
            input_ptrs,
            state_ptr,
            erased.state_drop_fn,
        ));

        output_idx
    }

    // -- Untyped helpers (for FFI / bridge) -----------------------------------

    /// Register a [`Source`], returning the output node index instead of a
    /// typed [`Handle`].
    pub fn add_source_untyped<S: Source>(&mut self, source: S) -> usize {
        self.add_source(source).index()
    }

    /// `TypeId` of a node's value.
    pub fn node_type_id(&self, index: usize) -> TypeId {
        self.graph.nodes[index].type_id
    }

    /// Raw value pointer for a node.
    pub(crate) fn node_value_ptr(&self, index: usize) -> *mut u8 {
        self.graph.nodes[index].value
    }

    /// Mutable reference to the closure state of a node.
    ///
    /// Returns `None` if the node has no closure.
    ///
    /// # Safety
    ///
    /// The caller must ensure the returned pointer is cast to the correct
    /// state type and not used after the node is dropped.
    pub(crate) fn closure_state_ptr(&self, index: usize) -> Option<*mut u8> {
        self.graph.nodes[index]
            .closure
            .as_ref()
            .map(|c| c.state)
    }

    // -- Flush ---------------------------------------------------------------

    /// Propagate updates through the DAG.
    pub fn flush(&mut self, timestamp: i64, updated_sources: &[usize]) {
        self.graph.flush(timestamp, updated_sources);
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

    // -- Erased operator test -------------------------------------------------

    #[test]
    fn scenario_add_erased_operator() {
        let mut sc = Scenario::new();
        let ha = sc.create_node(Array::scalar(0.0_f64));
        let hb = sc.create_node(Array::scalar(0.0_f64));

        let input_ptrs: Box<[*const u8]> = [ha.index(), hb.index()]
            .iter()
            .map(|&idx| sc.graph.nodes[idx].value as *const u8)
            .collect();
        // SAFETY: input pointers are valid Array<f64>.
        let erased = unsafe { ErasedOperator::from_operator(add::<f64>(), &input_ptrs, i64::MIN) };
        let out_idx = sc.add_erased_operator(erased, &[ha.index(), hb.index()], None);

        sc.value_mut(ha)[0] = 10.0;
        sc.value_mut(hb)[0] = 3.0;
        sc.flush(1, &[ha.index(), hb.index()]);

        let out = unsafe { &*(sc.graph.nodes[out_idx].value as *const Array<f64>) };
        assert_eq!(out.as_slice(), &[13.0]);
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
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2, 3], vec![10.0, 20.0, 30.0], 1));
        let hclock = sc.add_source(clock(vec![2]));

        let ho = sc.add_operator_periodic(Filter::new(|_: &Array<f64>| true), (ha,), hclock);
        let hs = sc.add_operator(Record::<f64>::new(), (ho,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hs);
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), &[2]);
        assert_eq!(series.values(), &[20.0]);
    }

    #[tokio::test]
    async fn scenario_periodic_two_inputs() {
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0], 1));
        let hb = sc.add_source(ArraySource::new(vec![1, 3], vec![10.0, 30.0], 1));
        let hclock = sc.add_source(clock(vec![2]));

        let ho = sc.add_operator_periodic(add::<f64>(), (ha, hb), hclock);
        let hs = sc.add_operator(Record::<f64>::new(), (ho,));

        sc.run().await;

        let series: &Series<f64> = sc.value(hs);
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), &[2]);
        assert_eq!(series.values(), &[12.0]);
    }

    #[tokio::test]
    async fn scenario_periodic_multiple_ticks() {
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
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[2, 4]);
        assert_eq!(series.values(), &[20.0, 40.0]);
    }
}
