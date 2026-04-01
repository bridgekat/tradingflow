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
//! Node indices encode topological order: if node `j` depends on node `i`,
//! then `i < j`.  Flush propagation uses a min-heap keyed by node index to
//! process updates in topological order.
//!
//! # Registration API
//!
//! - [`Scenario::add_const`] — register a constant node (shorthand for
//!   [`Const`](crate::operators::Const) operator).
//! - [`Scenario::add_source`] — register a [`Source`](crate::source::Source),
//!   creating an output node.  Requires a tokio runtime for sources that spawn
//!   tasks internally.
//! - [`Scenario::add_operator`] — register a concrete
//!   [`Operator`](crate::operator::Operator), creating its output node.
//!   Accepts typed [`Handle`]s for inputs and an optional trigger handle.
//!
//! All operator registration flows through [`Scenario::add_erased_operator`],
//! which accepts an [`ErasedOperator`](crate::operator::ErasedOperator).
//! Source registration flows through [`Scenario::add_erased_source`], which
//! accepts an [`ErasedSource`](crate::source::ErasedSource).
//!
//! # Execution
//!
//! - [`Scenario::flush`] — manually propagate updates through the DAG for a
//!   set of updated source node indices at a given timestamp.
//! - [`Scenario::run`] — async POCQ (Point-of-Coherency Queue) event loop
//!   that drains all historical and live source channels in timestamp order,
//!   coalescing same-timestamp events before flushing.  See the [`queue`]
//!   module for ordering guarantees and complexity.
//!
//! # Sub-modules
//!
//! - [`handle`] — [`Handle<T>`] typed index and [`InputTypesHandles`] trait.

mod graph;
pub mod handle;
mod node;
mod queue;

pub use handle::{Handle, InputTypesHandles};

use std::any::TypeId;

use crate::operator::{ErasedOperator, Operator};
use crate::operators::Const;
use crate::source::{ErasedSource, Source};

use graph::Graph;
use node::Node;

/// Type-erased DAG runtime for event-driven computation.
///
/// # Type-safe API example
///
/// ```
/// use tradingflow::{Scenario, Array};
/// use tradingflow::operators::Add;
///
/// let mut sc = Scenario::new();
///
/// let ha = sc.add_const(Array::scalar(0.0));
/// let hb = sc.add_const(Array::scalar(0.0));
/// let hc = sc.add_operator(Add::new(), (ha, hb), None);
///
/// sc.value_mut(ha)[0] = 10.0;
/// sc.value_mut(hb)[0] = 3.0;
/// sc.flush(1, &[ha.index(), hb.index()]);
///
/// assert_eq!(sc.value(hc).as_slice(), &[13.0]);
/// ```
pub struct Scenario {
    graph: Graph,
    source_indices: Vec<usize>,
}

impl Scenario {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            source_indices: Vec::new(),
        }
    }

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
        unsafe { &*(node.value_ptr as *const T) }
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
        unsafe { &mut *(node.value_ptr as *mut T) }
    }

    /// Raw value pointer to a node.
    pub fn value_ptr(&self, index: usize) -> *mut u8 {
        self.graph.nodes[index].value_ptr
    }

    /// Register a constant node with an initial value.
    pub fn add_const<T: Send + 'static>(&mut self, value: T) -> Handle<T> {
        self.add_operator(Const::new(value), (), None)
    }

    /// Register a [`Source`], creating the output node.
    ///
    /// Sources that use [`tokio::spawn`] internally (e.g. [`ArraySource`],
    /// [`IterSource`]) require a tokio runtime to be active.
    pub fn add_source<S: Source>(&mut self, source: S) -> Handle<S::Output> {
        let erased = ErasedSource::from_source(source);
        Handle::new(self.add_erased_source(erased))
    }

    /// Register an [`Operator`], creating its output node.
    pub fn add_operator<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputTypesHandles>::Handles>,
        trigger: Option<Handle<()>>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputTypesHandles,
    {
        let input_indices = <O::Inputs as InputTypesHandles>::node_indices(&inputs.into());
        let trigger_index = trigger.map(|c| c.index());
        let erased = ErasedOperator::from_operator(operator, input_indices.len());
        Handle::new(self.add_erased_operator(erased, &input_indices, trigger_index))
    }

    /// Register a type-erased source.
    pub fn add_erased_source(&mut self, erased: ErasedSource) -> usize {
        let node = Node::from_erased_source(erased, i64::MIN);
        let output_idx = self.graph.add_node(node);
        self.source_indices.push(output_idx);
        output_idx
    }

    /// Register a type-erased operator.
    pub fn add_erased_operator(
        &mut self,
        erased: ErasedOperator,
        input_indices: &[usize],
        trigger_index: Option<usize>,
    ) -> usize {
        for &idx in input_indices {
            assert!(
                idx < self.graph.len(),
                "invalid index: node {idx} out of range"
            );
        }
        let input_ptrs: Box<[*const u8]> = input_indices
            .iter()
            .map(|&idx| self.graph.nodes[idx].value_ptr as *const u8)
            .collect();
        let input_type_ids: Box<[TypeId]> = input_indices
            .iter()
            .map(|&idx| self.graph.nodes[idx].type_id)
            .collect();
        let node = Node::from_erased_operator(erased, input_ptrs, &input_type_ids, i64::MIN);
        let output_idx = self.graph.add_node(node);
        match trigger_index {
            None => {
                for &input_idx in input_indices {
                    self.graph.add_trigger_edge(input_idx, output_idx);
                }
            }
            Some(trigger_idx) => {
                self.graph.add_trigger_edge(trigger_idx, output_idx);
            }
        }
        output_idx
    }

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
    use crate::operators::{Add, Filter, Record};
    use crate::series::Series;
    use crate::sources::ArraySource;

    // -- Basic tests ----------------------------------------------------------

    #[test]
    fn scenario_arbitrary_type() {
        use std::collections::BTreeMap;

        let mut sc = Scenario::new();
        let h = sc.add_const(BTreeMap::<String, f64>::new());

        sc.value_mut(h).insert("price".to_string(), 42.0);
        assert_eq!(sc.value(h).get("price"), Some(&42.0));
    }

    // -- Simple operator tests ------------------------------------------------

    #[test]
    fn scenario_simple_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::scalar(0.0_f64));
        let hb = sc.add_const(Array::scalar(0.0_f64));
        let hc = sc.add_operator(Add::new(), (ha, hb), None);

        sc.value_mut(ha)[0] = 10.0;
        sc.value_mut(hb)[0] = 3.0;
        sc.flush(1, &[ha.index(), hb.index()]);

        assert_eq!(sc.value(hc).as_slice(), &[13.0]);
    }

    #[test]
    fn scenario_strided_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::from_vec(&[2], vec![1.0_f64, 2.0]));
        let hb = sc.add_const(Array::from_vec(&[2], vec![10.0_f64, 20.0]));
        let hc = sc.add_operator(Add::new(), (ha, hb), None);

        sc.flush(1, &[ha.index(), hb.index()]);
        assert_eq!(sc.value(hc).as_slice(), &[11.0, 22.0]);
    }

    #[test]
    fn scenario_chain() {
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::scalar(2.0_f64));
        let hb = sc.add_const(Array::scalar(3.0_f64));
        let hab = sc.add_operator(Add::new(), (ha, hb), None);

        use crate::operators::Multiply;
        let hout = sc.add_operator(Multiply::new(), (hab, ha), None);

        sc.flush(1, &[ha.index(), hb.index()]);
        // (2+3) * 2 = 10
        assert_eq!(sc.value(hout).as_slice(), &[10.0]);
    }

    #[test]
    fn scenario_record() {
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::scalar(0.0_f64));
        let hb = sc.add_const(Array::scalar(0.0_f64));
        let hsum = sc.add_operator(Add::new(), (ha, hb), None);
        let hseries = sc.add_operator(Record::<f64>::new(), (hsum,), None);

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

    // -- Async run tests ------------------------------------------------------

    #[tokio::test]
    async fn scenario_run_single_source() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 2, 3], vec![10.0, 20.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hseries = sc.add_operator(Record::<f64>::new(), (ha,), None);

        sc.run().await;

        let series: &Series<f64> = sc.value(hseries);
        assert_eq!(series.len(), 3);
        assert_eq!(series.timestamps(), &[1, 2, 3]);
        assert_eq!(series.values(), &[10.0, 20.0, 30.0]);
    }

    #[tokio::test]
    async fn scenario_run_two_sources_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 3], vec![10.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![2, 3], vec![20.0, 40.0]),
            Array::scalar(0.0),
        ));
        let ho = sc.add_operator(Add::new(), (ha, hb), None);
        let hseries = sc.add_operator(Record::<f64>::new(), (ho,), None);

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
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 2], vec![10.0, 20.0]),
            Array::scalar(0.0),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 2], vec![100.0, 200.0]),
            Array::scalar(0.0),
        ));
        let ho = sc.add_operator(Add::new(), (ha, hb), None);
        let hseries = sc.add_operator(Record::<f64>::new(), (ho,), None);

        sc.run().await;

        let series: &Series<f64> = sc.value(hseries);
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[1, 2]);
        assert_eq!(series.values(), &[110.0, 220.0]);
    }

    #[tokio::test]
    async fn scenario_run_chained() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 2], vec![2.0, 5.0]),
            Array::scalar(0.0),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 2], vec![3.0, 10.0]),
            Array::scalar(0.0),
        ));
        let hab = sc.add_operator(Add::new(), (ha, hb), None);

        use crate::operators::Multiply;
        let hout = sc.add_operator(Multiply::new(), (hab, ha), None);
        let hseries = sc.add_operator(Record::<f64>::new(), (hout,), None);

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
            Series::from_vec(&[], vec![1, 2, 3, 4], vec![1.0, 5.0, 2.0, 10.0]),
            Array::scalar(0.0),
        ));
        let ho = sc.add_operator(Filter::new(|v: &Array<f64>| v[0] > 3.0), (ha,), None);
        let hseries = sc.add_operator(Record::<f64>::new(), (ho,), None);

        sc.run().await;

        let series: &Series<f64> = sc.value(hseries);
        // passes: ts=2(5.0), ts=4(10.0)
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[2, 4]);
        assert_eq!(series.values(), &[5.0, 10.0]);
    }

    #[tokio::test]
    async fn scenario_run_periodic_single_input() {
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 2, 3], vec![10.0, 20.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hclock = sc.add_source(clock(vec![2]));

        let ho = sc.add_operator(Filter::new(|_: &Array<f64>| true), (ha,), Some(hclock));
        let hs = sc.add_operator(Record::<f64>::new(), (ho,), None);

        sc.run().await;

        let series: &Series<f64> = sc.value(hs);
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), &[2]);
        assert_eq!(series.values(), &[20.0]);
    }

    #[tokio::test]
    async fn scenario_run_periodic_two_inputs() {
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 2, 3], vec![1.0, 2.0, 3.0]),
            Array::scalar(0.0),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 3], vec![10.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hclock = sc.add_source(clock(vec![2]));

        let ho = sc.add_operator(Add::new(), (ha, hb), Some(hclock));
        let hs = sc.add_operator(Record::<f64>::new(), (ho,), None);

        sc.run().await;

        let series: &Series<f64> = sc.value(hs);
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), &[2]);
        assert_eq!(series.values(), &[12.0]);
    }

    #[tokio::test]
    async fn scenario_run_periodic_multiple_ticks() {
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], vec![1, 2, 3, 4, 5], vec![10.0, 20.0, 30.0, 40.0, 50.0]),
            Array::scalar(0.0),
        ));
        let hclock = sc.add_source(clock(vec![2, 4]));

        let ho = sc.add_operator(Filter::new(|_: &Array<f64>| true), (ha,), Some(hclock));
        let hs = sc.add_operator(Record::<f64>::new(), (ho,), None);

        sc.run().await;

        let series: &Series<f64> = sc.value(hs);
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), &[2, 4]);
        assert_eq!(series.values(), &[20.0, 40.0]);
    }
}
