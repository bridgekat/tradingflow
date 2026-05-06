//! Scenario — a pure definition of an event-driven computation graph.
//!
//! A [`Scenario`] is an *immutable description* of a directed acyclic graph
//! of nodes, where each node is either a [`Source`](crate::source::Source)
//! producing timestamped events or an [`Operator`](crate::operator::Operator)
//! consuming upstream values to produce its own.  The scenario itself owns
//! no node values, channel receivers, or operator state — it stores only
//! the type-erased descriptors ([`ErasedSource`](crate::source::ErasedSource),
//! [`ErasedOperator`](crate::operator::ErasedOperator)) and the wiring
//! between them.
//!
//! All runtime state lives on a [`Session`], built from a scenario by
//! [`Scenario::build_session`].  Every call replays each descriptor's `init`
//! against fresh buffers, so a single scenario can drive any number of
//! independent sessions — useful for parameter sweeps, repeated
//! backtests, and reproducibility.
//!
//! # Architecture
//!
//! Internally, [`Session`] nodes are stored as type-erased
//! `(pointer, TypeId)` slots.  Type safety is enforced at registration
//! time on the scenario via [`Handle<T>`] and [`TypeId`] checks.  After
//! registration, operator dispatch uses raw pointer casts through
//! monomorphised function pointers — zero dynamic dispatch overhead on
//! the hot path.
//!
//! Node indices encode topological order: if node `j` depends on node `i`,
//! then `i < j`.  Flush propagation uses a min-heap keyed by node index to
//! process updates in topological order.
//!
//! # Registration API
//!
//! - [`Scenario::add_const`] — register a constant node (shorthand for
//!   [`Const`](crate::operators::Const) operator).
//! - [`Scenario::add_source`] — register a [`Source`](crate::source::Source).
//! - [`Scenario::add_operator`] — register a concrete
//!   [`Operator`](crate::operator::Operator).  Accepts typed [`Handle`]s
//!   for inputs.
//!
//! All operator registration flows through [`Scenario::add_erased_operator`],
//! and source registration through [`Scenario::add_erased_source`].
//!
//! # Execution
//!
//! - [`Scenario::build_session`] — construct a fresh [`Session`] by
//!   replaying every registered descriptor's `init` against newly
//!   allocated buffers.
//! - [`Scenario::run`] — convenience: build a session and drive its
//!   async event loop until every source is exhausted, returning the
//!   populated session for inspection.
//!
//! Per-session manual stepping ([`Session::flush`]) and the lower-level
//! event loop ([`Session::run`], [`Session::run_with_shutdown`]) live on
//! [`Session`].
//!
//! # Sub-modules
//!
//! - [`handle`] — [`Handle<T>`] typed index and [`InputTypesHandles`] trait.

mod graph;
pub mod handle;
mod node;
mod queue;
mod session;

pub use handle::{Handle, InputTypesHandles};
pub use queue::ShutdownFlag;
pub use session::Session;

use std::any::TypeId;

use crate::Instant;
use crate::operator::{ErasedOperator, Operator};
use crate::operators::Const;
use crate::source::{ErasedSource, Source};

use graph::Graph;
use node::Node;

/// One entry in a [`Scenario`] definition: either a source descriptor or
/// an operator descriptor with its upstream input indices.
enum NodeDescriptor {
    Source(ErasedSource),
    Operator {
        erased: ErasedOperator,
        input_indices: Box<[usize]>,
    },
}

impl NodeDescriptor {
    fn output_type_id(&self) -> TypeId {
        match self {
            NodeDescriptor::Source(s) => s.output_type_id(),
            NodeDescriptor::Operator { erased, .. } => erased.output_type_id(),
        }
    }
}

/// Pure definition of a computation graph.
///
/// Holds [`ErasedSource`] / [`ErasedOperator`] descriptors and the input
/// wiring between them.  Carries no per-run state — all runtime state
/// (value buffers, channel receivers, operator state) lives on a
/// [`Session`] built via [`build_session`](Self::build_session).
///
/// # Type-safe API example
///
/// ```
/// use tradingflow::{Scenario, Array};
/// use tradingflow::operators::num::Add;
///
/// use tradingflow::Instant;
///
/// let mut sc = Scenario::new();
///
/// let ha = sc.add_const(Array::scalar(0.0));
/// let hb = sc.add_const(Array::scalar(0.0));
/// let hc = sc.add_operator(Add::new(), (ha, hb));
///
/// let mut session = sc.build_session();
/// session.value_mut(ha)[0] = 10.0;
/// session.value_mut(hb)[0] = 3.0;
/// session.flush(Instant::from_nanos(1), &[ha.index(), hb.index()]);
///
/// assert_eq!(session.value(hc).as_slice(), &[13.0]);
/// ```
pub struct Scenario {
    /// Per-node descriptor in declaration (= topological) order.  Source
    /// indices are derived from this on demand by
    /// [`build_session`](Self::build_session) — no need to maintain a
    /// separate vector.
    descriptors: Vec<NodeDescriptor>,
    /// Cumulative estimated event count across all registered sources.
    /// Updated incrementally in [`Scenario::add_erased_source`]; becomes
    /// `None` and stays `None` as soon as any source reports `None`.
    estimated_event_count: Option<usize>,
}

impl Scenario {
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
            estimated_event_count: Some(0),
        }
    }

    /// Number of registered nodes.
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Whether any nodes are registered.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// The output `TypeId` of a registered node.  Used by the Python
    /// bridge to build the type-id list for operators with `!Sized`
    /// `Inputs` (e.g. Stack/Concat).
    #[cfg(feature = "python")]
    pub(crate) fn node_type_id(&self, index: usize) -> TypeId {
        self.descriptors[index].output_type_id()
    }

    /// Register a constant node with an initial value.
    pub fn add_const<T: Clone + Send + 'static>(&mut self, value: T) -> Handle<T> {
        // `Clone` is required for the underlying `Const` operator: the
        // erased layer clones its spec on every init, and the value is
        // moved into the freshly built output.
        self.add_operator(Const::new(value), ())
    }

    /// Register a [`Source`], creating the output node.
    ///
    /// Sources that use [`tokio::spawn`] internally (e.g. [`ArraySource`],
    /// [`IterSource`]) require a tokio runtime to be active when the
    /// resulting session's event loop runs — registration itself does not
    /// touch tokio.
    pub fn add_source<S: Source>(&mut self, source: S) -> Handle<S::Output> {
        let erased = ErasedSource::from_source(source);
        Handle::new(self.add_erased_source(erased))
    }

    /// Register an [`Operator`], creating its output node.
    pub fn add_operator<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputTypesHandles>::Handles>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputTypesHandles,
    {
        let handles = inputs.into();
        let arity = <O::Inputs as InputTypesHandles>::arity(&handles);
        let mut input_indices: Vec<usize> = vec![0usize; arity];
        {
            let mut writer = crate::data::FlatWrite::new(&mut input_indices);
            <O::Inputs as InputTypesHandles>::write_node_indices(&handles, &mut writer);
        }

        // Pre-size the type-id buffer using the handles arity (accounts for
        // runtime slice lengths).  Then call type_ids_to_flat — for Sized
        // inputs it fills the whole buffer; for a trailing [T] slice it
        // fills the remaining space using the buffer length as the count.
        let mut type_ids = vec![std::any::TypeId::of::<()>(); arity];
        {
            let mut writer = crate::data::FlatWrite::new(&mut type_ids);
            <O::Inputs as crate::data::InputTypes>::type_ids_to_flat(&mut writer);
        }
        let erased =
            ErasedOperator::from_operator_with_type_ids(operator, type_ids.into_boxed_slice());
        Handle::new(self.add_erased_operator(erased, &input_indices))
    }

    /// Register a type-erased source.
    pub fn add_erased_source(&mut self, erased: ErasedSource) -> usize {
        let estimate = erased.estimated_event_count();
        self.estimated_event_count = match (self.estimated_event_count, estimate) {
            (Some(acc), Some(n)) => Some(acc.saturating_add(n)),
            _ => None,
        };
        let idx = self.descriptors.len();
        self.descriptors.push(NodeDescriptor::Source(erased));
        idx
    }

    /// Register a type-erased operator.
    ///
    /// Validates that every input node already exists, and that its
    /// declared input type-ids match the upstream nodes' output type-ids.
    /// Panics on arity or TypeId mismatch — same diagnostics as the
    /// previous in-place graph build, just at registration time against
    /// the descriptor list.
    pub fn add_erased_operator(
        &mut self,
        erased: ErasedOperator,
        input_indices: &[usize],
    ) -> usize {
        // Validate inputs exist and types match before storing.
        let expected_input_type_ids = erased.input_type_ids();
        assert_eq!(
            expected_input_type_ids.len(),
            input_indices.len(),
            "arity mismatch: operator expects {} inputs, got {}",
            expected_input_type_ids.len(),
            input_indices.len(),
        );
        for (i, &idx) in input_indices.iter().enumerate() {
            assert!(
                idx < self.descriptors.len(),
                "invalid index: node {idx} out of range",
            );
            let actual = self.descriptors[idx].output_type_id();
            assert_eq!(
                expected_input_type_ids[i], actual,
                "type mismatch at input {i}",
            );
        }

        let idx = self.descriptors.len();
        self.descriptors.push(NodeDescriptor::Operator {
            erased,
            input_indices: input_indices.to_vec().into_boxed_slice(),
        });
        idx
    }

    /// Sum of estimated event counts across all sources.
    ///
    /// Returns `Some(total)` only when **every** registered source provides
    /// an estimate; otherwise `None`.  Cached — updated incrementally as
    /// sources are registered.  Used by [`Session::run`] for progress
    /// reporting.
    #[inline]
    pub fn estimated_event_count(&self) -> Option<usize> {
        self.estimated_event_count
    }

    /// Build a fresh [`Session`] by replaying every registered descriptor.
    ///
    /// Each call allocates fresh node buffers and operator state via the
    /// descriptors' [`init`](crate::source::ErasedSource::init) closures
    /// (which take `&self` and clone the captured spec on every call).
    /// The returned session is independent of any other previously built
    /// from the same scenario.
    pub fn build_session(&self) -> Session {
        let mut graph = Graph::new();
        let mut source_indices = Vec::new();

        for (idx, descriptor) in self.descriptors.iter().enumerate() {
            match descriptor {
                NodeDescriptor::Source(erased) => {
                    let node = Node::from_erased_source(erased, Instant::MIN);
                    graph.add_node(node);
                    source_indices.push(idx);
                }
                NodeDescriptor::Operator {
                    erased,
                    input_indices,
                } => {
                    let input_ptrs: Box<[*const u8]> = input_indices
                        .iter()
                        .map(|&i| graph.nodes[i].value_ptr as *const u8)
                        .collect();
                    let input_type_ids: Box<[TypeId]> = input_indices
                        .iter()
                        .map(|&i| graph.nodes[i].type_id)
                        .collect();
                    let input_node_indices: Box<[usize]> = input_indices.clone();
                    let node = Node::from_erased_operator(
                        erased,
                        input_ptrs,
                        input_node_indices,
                        &input_type_ids,
                        Instant::MIN,
                    );
                    let output_idx = graph.add_node(node);
                    for (pos, &input_idx) in input_indices.iter().enumerate() {
                        graph.add_trigger_edge(input_idx, output_idx, pos);
                    }
                }
            }
        }

        Session {
            graph,
            source_indices,
            estimated_event_count: self.estimated_event_count,
        }
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
    use crate::Array;
    use crate::Series;
    use crate::operators::num::Add;
    use crate::operators::{Filter, Record};
    use crate::sources::ArraySource;

    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    fn tss(xs: &[i64]) -> Vec<Instant> {
        xs.iter().copied().map(Instant::from_nanos).collect()
    }

    // -- Basic tests ----------------------------------------------------------

    #[test]
    fn scenario_arbitrary_type() {
        use std::collections::BTreeMap;

        let mut sc = Scenario::new();
        let h = sc.add_const(BTreeMap::<String, f64>::new());

        let mut session = sc.build_session();
        session.value_mut(h).insert("price".to_string(), 42.0);
        assert_eq!(session.value(h).get("price"), Some(&42.0));
    }

    // -- Simple operator tests ------------------------------------------------

    #[test]
    fn scenario_simple_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::scalar(0.0_f64));
        let hb = sc.add_const(Array::scalar(0.0_f64));
        let hc = sc.add_operator(Add::new(), (ha, hb));

        let mut session = sc.build_session();
        session.value_mut(ha)[0] = 10.0;
        session.value_mut(hb)[0] = 3.0;
        session.flush(ts(1), &[ha.index(), hb.index()]);

        assert_eq!(session.value(hc).as_slice(), &[13.0]);
    }

    #[test]
    fn scenario_strided_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::from_vec(&[2], vec![1.0_f64, 2.0]));
        let hb = sc.add_const(Array::from_vec(&[2], vec![10.0_f64, 20.0]));
        let hc = sc.add_operator(Add::new(), (ha, hb));

        let mut session = sc.build_session();
        session.flush(ts(1), &[ha.index(), hb.index()]);
        assert_eq!(session.value(hc).as_slice(), &[11.0, 22.0]);
    }

    #[test]
    fn scenario_chain() {
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::scalar(2.0_f64));
        let hb = sc.add_const(Array::scalar(3.0_f64));
        let hab = sc.add_operator(Add::new(), (ha, hb));

        use crate::operators::num::Multiply;
        let hout = sc.add_operator(Multiply::new(), (hab, ha));

        let mut session = sc.build_session();
        session.flush(ts(1), &[ha.index(), hb.index()]);
        // (2+3) * 2 = 10
        assert_eq!(session.value(hout).as_slice(), &[10.0]);
    }

    #[test]
    fn scenario_record() {
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::scalar(0.0_f64));
        let hb = sc.add_const(Array::scalar(0.0_f64));
        let hsum = sc.add_operator(Add::new(), (ha, hb));
        let hseries = sc.add_operator(Record::<f64>::new(), hsum);

        let mut session = sc.build_session();
        session.value_mut(ha)[0] = 10.0;
        session.value_mut(hb)[0] = 3.0;
        session.flush(ts(1), &[ha.index(), hb.index()]);

        session.value_mut(ha)[0] = 20.0;
        session.value_mut(hb)[0] = 7.0;
        session.flush(ts(2), &[ha.index(), hb.index()]);

        let series: &Series<f64> = session.value(hseries);
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), tss(&[1, 2]).as_slice());
        assert_eq!(series.values(), &[13.0, 27.0]);
    }

    #[test]
    fn scenario_reusable_definition() {
        // The same scenario can drive multiple independent sessions.
        let mut sc = Scenario::new();
        let ha = sc.add_const(Array::scalar(0.0_f64));
        let hb = sc.add_const(Array::scalar(0.0_f64));
        let hc = sc.add_operator(Add::new(), (ha, hb));

        let mut s1 = sc.build_session();
        s1.value_mut(ha)[0] = 10.0;
        s1.value_mut(hb)[0] = 3.0;
        s1.flush(ts(1), &[ha.index(), hb.index()]);
        assert_eq!(s1.value(hc).as_slice(), &[13.0]);

        let mut s2 = sc.build_session();
        // Fresh session — no state carried over from s1.
        assert_eq!(s2.value(hc).as_slice(), &[0.0]);
        s2.value_mut(ha)[0] = 100.0;
        s2.value_mut(hb)[0] = 200.0;
        s2.flush(ts(1), &[ha.index(), hb.index()]);
        assert_eq!(s2.value(hc).as_slice(), &[300.0]);

        // s1 still has its own state.
        assert_eq!(s1.value(hc).as_slice(), &[13.0]);
    }

    // -- Async run tests ------------------------------------------------------

    #[tokio::test]
    async fn scenario_run_single_source() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3]), vec![10.0, 20.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hseries = sc.add_operator(Record::<f64>::new(), ha);

        let session = sc.run(|_, _, _| {}).await;

        let series: &Series<f64> = session.value(hseries);
        assert_eq!(series.len(), 3);
        assert_eq!(series.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(series.values(), &[10.0, 20.0, 30.0]);
    }

    #[tokio::test]
    async fn scenario_run_two_sources_add() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 3]), vec![10.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[2, 3]), vec![20.0, 40.0]),
            Array::scalar(0.0),
        ));
        let ho = sc.add_operator(Add::new(), (ha, hb));
        let hseries = sc.add_operator(Record::<f64>::new(), ho);

        let session = sc.run(|_, _, _| {}).await;

        let series: &Series<f64> = session.value(hseries);
        // ts=1: 10+0=10, ts=2: 10+20=30, ts=3: 30+40=70
        assert_eq!(series.len(), 3);
        assert_eq!(series.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(series.values(), &[10.0, 30.0, 70.0]);
    }

    #[tokio::test]
    async fn scenario_run_coalescing() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2]), vec![10.0, 20.0]),
            Array::scalar(0.0),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2]), vec![100.0, 200.0]),
            Array::scalar(0.0),
        ));
        let ho = sc.add_operator(Add::new(), (ha, hb));
        let hseries = sc.add_operator(Record::<f64>::new(), ho);

        let session = sc.run(|_, _, _| {}).await;

        let series: &Series<f64> = session.value(hseries);
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), tss(&[1, 2]).as_slice());
        assert_eq!(series.values(), &[110.0, 220.0]);
    }

    #[tokio::test]
    async fn scenario_run_chained() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2]), vec![2.0, 5.0]),
            Array::scalar(0.0),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2]), vec![3.0, 10.0]),
            Array::scalar(0.0),
        ));
        let hab = sc.add_operator(Add::new(), (ha, hb));

        use crate::operators::num::Multiply;
        let hout = sc.add_operator(Multiply::new(), (hab, ha));
        let hseries = sc.add_operator(Record::<f64>::new(), hout);

        let session = sc.run(|_, _, _| {}).await;

        let series: &Series<f64> = session.value(hseries);
        assert_eq!(series.len(), 2);
        // ts=1: (2+3)*2=10, ts=2: (5+10)*5=75
        assert_eq!(series.values(), &[10.0, 75.0]);
    }

    #[tokio::test]
    async fn scenario_run_filter() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3, 4]), vec![1.0, 5.0, 2.0, 10.0]),
            Array::scalar(0.0),
        ));
        let ho = sc.add_operator(Filter::new(|v: &Array<f64>| v[0] > 3.0), ha);
        let hseries = sc.add_operator(Record::<f64>::new(), ho);

        let session = sc.run(|_, _, _| {}).await;

        let series: &Series<f64> = session.value(hseries);
        // passes: ts=2(5.0), ts=4(10.0)
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), tss(&[2, 4]).as_slice());
        assert_eq!(series.values(), &[5.0, 10.0]);
    }

    #[tokio::test]
    async fn scenario_run_periodic_single_input() {
        use crate::operators::Clocked;
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3]), vec![10.0, 20.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hclock = sc.add_source(clock(tss(&[2])));

        let ho = sc.add_operator(
            Clocked::new(Filter::new(|_: &Array<f64>| true)),
            (hclock, ha),
        );
        let hs = sc.add_operator(Record::<f64>::new(), ho);

        let session = sc.run(|_, _, _| {}).await;

        let series: &Series<f64> = session.value(hs);
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), tss(&[2]).as_slice());
        assert_eq!(series.values(), &[20.0]);
    }

    #[tokio::test]
    async fn scenario_run_periodic_two_inputs() {
        use crate::operators::Clocked;
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3]), vec![1.0, 2.0, 3.0]),
            Array::scalar(0.0),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 3]), vec![10.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hclock = sc.add_source(clock(tss(&[2])));

        let ho = sc.add_operator(Clocked::new(Add::new()), (hclock, (ha, hb)));
        let hs = sc.add_operator(Record::<f64>::new(), ho);

        let session = sc.run(|_, _, _| {}).await;

        let series: &Series<f64> = session.value(hs);
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), tss(&[2]).as_slice());
        assert_eq!(series.values(), &[12.0]);
    }

    #[tokio::test]
    async fn scenario_run_periodic_multiple_ticks() {
        use crate::operators::Clocked;
        use crate::sources::clock;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(
                &[],
                tss(&[1, 2, 3, 4, 5]),
                vec![10.0, 20.0, 30.0, 40.0, 50.0],
            ),
            Array::scalar(0.0),
        ));
        let hclock = sc.add_source(clock(tss(&[2, 4])));

        let ho = sc.add_operator(
            Clocked::new(Filter::new(|_: &Array<f64>| true)),
            (hclock, ha),
        );
        let hs = sc.add_operator(Record::<f64>::new(), ho);

        let session = sc.run(|_, _, _| {}).await;

        let series: &Series<f64> = session.value(hs);
        assert_eq!(series.len(), 2);
        assert_eq!(series.timestamps(), tss(&[2, 4]).as_slice());
        assert_eq!(series.values(), &[20.0, 40.0]);
    }

    #[tokio::test]
    async fn scenario_repeated_run() {
        // The same scenario definition can drive multiple independent
        // event-loop sessions.  Each session is built fresh from the
        // descriptors — no state carries over.
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3]), vec![10.0, 20.0, 30.0]),
            Array::scalar(0.0),
        ));
        let hseries = sc.add_operator(Record::<f64>::new(), ha);

        for _ in 0..3 {
            let session = sc.run(|_, _, _| {}).await;
            let series: &Series<f64> = session.value(hseries);
            assert_eq!(series.len(), 3);
            assert_eq!(series.timestamps(), tss(&[1, 2, 3]).as_slice());
            assert_eq!(series.values(), &[10.0, 20.0, 30.0]);
        }
    }
}
