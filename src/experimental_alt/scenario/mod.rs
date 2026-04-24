//! Wavefront runtime — [`Scenario`], [`Handle`], and internals.
//!
//! Public facade mirrors the existing [`crate::scenario::Scenario`] where
//! practical — `new`, `add_const`, `add_source`, `add_operator`,
//! `add_erased_source`, `add_erased_operator_typed`, `value`, `run`,
//! `run_with_shutdown` — plus [`Scenario::seal`], which finalises graph
//! bookkeeping before a run.
//!
//! # Sub-modules
//!
//! - [`handle`] — [`Handle<T>`] and [`InputTypesHandles`].
//! - [`node`] — [`Node`](node::Node), [`OutputStore`](node::OutputStore),
//!   per-tick bookkeeping atomics.
//! - [`graph`] — [`Graph`](graph::Graph) owns [`Node`]s and the
//!   trigger-edge index.
//! - [`scheduler`] — crossbeam-deque worker pool and readiness predicate.
//! - [`ingest`] — adapted source-merge loop.

pub mod graph;
pub mod handle;
pub mod ingest;
pub mod node;
pub mod scheduler;

pub use handle::{Handle, InputTypesHandles};

use std::any::TypeId;
use std::sync::Arc;

use super::data::{FlatWrite, InputTypes};
use super::operator::{ErasedOperator, Operator};
use super::source::{ErasedSource, Source};

use graph::Graph;

/// Type-erased wavefront computation graph.
pub struct Scenario {
    pub(crate) graph: Arc<Graph>,
    pipeline_width: usize,
    default_queue_cap: usize,
    sealed: bool,
}

impl Scenario {
    /// Create a fresh scenario.
    pub fn new() -> Self {
        Self {
            graph: Arc::new(Graph::new()),
            pipeline_width: 8,
            default_queue_cap: 8,
            sealed: false,
        }
    }

    /// Configure the pipeline width (maximum in-flight ticks).  Must be
    /// called before any node is registered.  Default 8.
    pub fn with_pipeline_width(mut self, w: usize) -> Self {
        assert!(!self.sealed, "cannot reconfigure a sealed scenario");
        assert!(
            self.graph.num_nodes() == 0,
            "pipeline width must be set before registering nodes",
        );
        assert!(w >= 1);
        self.pipeline_width = w;
        self.default_queue_cap = w.max(self.default_queue_cap);
        self
    }

    pub fn pipeline_width(&self) -> usize {
        self.pipeline_width
    }

    fn graph_mut(&mut self) -> &mut Graph {
        Arc::get_mut(&mut self.graph)
            .expect("cannot register nodes on a scenario that is being run")
    }

    /// Register a constant node — a 0-input operator whose output is the
    /// supplied value.  Never scheduled after init.
    pub fn add_const<T>(&mut self, value: T) -> Handle<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        self.add_operator(super::operators::r#const::Const::new(value), ())
    }

    /// Register a [`Source`].
    pub fn add_source<S: Source>(&mut self, source: S) -> Handle<S::Output>
    where
        S::Output: Clone + Send + Sync + 'static,
    {
        let erased = ErasedSource::from_source(source);
        Handle::new(self.add_erased_source::<S::Output>(erased))
    }

    /// Register an [`Operator`].
    pub fn add_operator<O: Operator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputTypesHandles>::Handles>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputTypesHandles,
        O::Output: Clone + Send + Sync + 'static,
    {
        let handles = inputs.into();
        let arity = <O::Inputs as InputTypesHandles>::arity(&handles);
        let mut input_indices: Vec<usize> = vec![0usize; arity];
        {
            let mut writer = crate::data::FlatWrite::new(&mut input_indices);
            <O::Inputs as InputTypesHandles>::write_node_indices(&handles, &mut writer);
        }
        let mut type_ids = vec![TypeId::of::<()>(); arity];
        {
            let mut writer = FlatWrite::new(&mut type_ids);
            <O::Inputs as InputTypes>::type_ids_to_flat(&mut writer);
        }
        let erased = ErasedOperator::from_operator_with_type_ids(
            operator,
            type_ids.into_boxed_slice(),
        );
        Handle::new(self.add_erased_operator_typed::<O::Output>(erased, &input_indices))
    }

    /// Register a type-erased [`Source`].
    pub fn add_erased_source<T>(&mut self, erased: ErasedSource) -> usize
    where
        T: Clone + Send + Sync + 'static,
    {
        assert_eq!(
            erased.output_type_id(),
            TypeId::of::<T>(),
            "add_erased_source: output type id mismatch",
        );
        let queue_cap = self.default_queue_cap;
        let width = self.pipeline_width;
        self.graph_mut().add_source::<T>(erased, queue_cap, width)
    }

    /// Register a type-erased [`Operator`] whose output type `T` is
    /// specified statically for storage allocation.
    pub fn add_erased_operator_typed<T>(
        &mut self,
        erased: ErasedOperator,
        input_indices: &[usize],
    ) -> usize
    where
        T: Clone + Send + Sync + 'static,
    {
        assert_eq!(
            erased.output_type_id(),
            TypeId::of::<T>(),
            "add_erased_operator_typed: output type id mismatch",
        );
        let queue_cap = self.default_queue_cap;
        let width = self.pipeline_width;
        self.graph_mut()
            .add_operator::<T>(erased, input_indices, queue_cap, width)
    }

    /// Latest committed value of a node.  Only safe to call after
    /// [`run`](Self::run) has returned (no concurrent writer).
    pub fn value<T: Clone + Send + Sync + 'static>(&self, h: Handle<T>) -> Option<Arc<T>> {
        self.graph.value::<T>(h.index())
    }

    /// Finalise graph bookkeeping.  Automatically called on first
    /// [`run`](Self::run) if not called explicitly.
    pub fn seal(&mut self) {
        if !self.sealed {
            self.graph_mut().seal();
            self.sealed = true;
        }
    }

    /// Run the ingest loop and the scheduler to completion.
    ///
    /// Must be called from within a tokio runtime context (the source
    /// ingest uses tokio mpsc channels).  Blocks until every source has
    /// closed and every scheduled task has completed.
    pub async fn run(&mut self) {
        self.run_with_shutdown(scheduler::ShutdownFlag::new()).await
    }

    /// Like [`run`](Self::run) but with a shared cooperative shutdown flag.
    pub async fn run_with_shutdown(&mut self, shutdown: scheduler::ShutdownFlag) {
        self.seal();
        scheduler::drive(Arc::clone(&self.graph), shutdown).await;
    }
}

impl Default for Scenario {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Basic end-to-end smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experimental_alt::data::{Array, Instant, Series};
    use crate::experimental_alt::operators::{Add, Clocked, Filter, Lag, Record, RollingMean};
    use crate::experimental_alt::sources::{ArraySource, clock};

    fn tss(xs: &[i64]) -> Vec<Instant> {
        xs.iter().copied().map(Instant::from_nanos).collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_source_record() {
        let mut sc = Scenario::new();
        let hs = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3]), vec![10.0_f64, 20.0, 30.0]),
            Array::scalar(0.0_f64),
        ));
        let hrec = sc.add_operator(Record::<f64>::new(), hs);

        sc.run().await;

        let series = sc.value(hrec).expect("record output exists");
        assert_eq!(series.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(series.values(), &[10.0, 20.0, 30.0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_plus_const_record() {
        let mut sc = Scenario::new();
        let hs = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3]), vec![10.0_f64, 20.0, 30.0]),
            Array::scalar(0.0_f64),
        ));
        let hk = sc.add_const(Array::scalar(5.0_f64));
        let hsum = sc.add_operator(Add::<f64>::new(), (hs, hk));
        let hrec = sc.add_operator(Record::<f64>::new(), hsum);

        sc.run().await;

        let series = sc.value(hrec).expect("record output exists");
        assert_eq!(series.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(series.values(), &[15.0, 25.0, 35.0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rolling_mean_through_record() {
        let mut sc = Scenario::new();
        let hs = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3, 4, 5]), vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]),
            Array::scalar(0.0_f64),
        ));
        let hrec = sc.add_operator(Record::<f64>::new(), hs);
        let hroll = sc.add_operator(RollingMean::<f64>::count(3), hrec);
        let hroll_rec = sc.add_operator(Record::<f64>::new(), hroll);

        sc.run().await;

        // RollingMean(3) over [10,20,30,40,50]:
        //   t=1: window size 1, no emit.
        //   t=2: window size 2, no emit.
        //   t=3: (10+20+30)/3 = 20.
        //   t=4: (20+30+40)/3 = 30.
        //   t=5: (30+40+50)/3 = 40.
        let series = sc.value(hroll_rec).expect("rolling output exists");
        assert_eq!(series.timestamps(), tss(&[3, 4, 5]).as_slice());
        assert_eq!(series.values(), &[20.0, 30.0, 40.0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lag_through_series() {
        let mut sc = Scenario::new();
        let hs = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3, 4]), vec![10.0_f64, 20.0, 30.0, 40.0]),
            Array::scalar(0.0_f64),
        ));
        let hrec = sc.add_operator(Record::<f64>::new(), hs);
        let hlag = sc.add_operator(Lag::<f64>::new(1, f64::NAN), hrec);
        let hlag_rec = sc.add_operator(Record::<f64>::new(), hlag);

        sc.run().await;

        let series = sc.value(hlag_rec).expect("lag output exists");
        // Lag(1) emits on every tick (returns true), with the value
        // from 1 step ago (NaN until t=2).
        assert_eq!(series.timestamps(), tss(&[1, 2, 3, 4]).as_slice());
        assert!(series.values()[0].is_nan());
        assert_eq!(&series.values()[1..], &[10.0, 20.0, 30.0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn filter_gates_downstream() {
        let mut sc = Scenario::new();
        let hs = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3, 4]), vec![1.0_f64, 5.0, 2.0, 10.0]),
            Array::scalar(0.0_f64),
        ));
        let hf = sc.add_operator(Filter::new(|v: &Array<f64>| v[0] > 3.0), hs);
        let hrec = sc.add_operator(Record::<f64>::new(), hf);

        sc.run().await;

        let series = sc.value(hrec).expect("filter output exists");
        // Only values > 3.0 reach Record: 5.0 at t=2, 10.0 at t=4.
        assert_eq!(series.timestamps(), tss(&[2, 4]).as_slice());
        assert_eq!(series.values(), &[5.0, 10.0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn clocked_filter_gates_on_clock() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3]), vec![10.0_f64, 20.0, 30.0]),
            Array::scalar(0.0_f64),
        ));
        let hclock = sc.add_source(clock(tss(&[2])));
        let ho = sc.add_operator(
            Clocked::<Filter<f64, _>, ()>::new(Filter::new(|_: &Array<f64>| true)),
            (hclock, ha),
        );
        let hrec = sc.add_operator(Record::<f64>::new(), ho);

        sc.run().await;

        let series = sc.value(hrec).expect("clocked output exists");
        // Only t=2 (when clock fires) produces output.
        assert_eq!(series.timestamps(), tss(&[2]).as_slice());
        assert_eq!(series.values(), &[20.0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn wide_fanout_all_branches_produce() {
        // One source feeds 8 independent Add(src, const_i) branches.
        // Verifies node-axis parallelism does not break correctness.
        let mut sc = Scenario::new();
        let hs = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 2, 3]), vec![100.0_f64, 200.0, 300.0]),
            Array::scalar(0.0_f64),
        ));
        let mut recs = Vec::new();
        for i in 0..8 {
            let hk = sc.add_const(Array::scalar(i as f64));
            let hsum = sc.add_operator(Add::<f64>::new(), (hs, hk));
            let hrec = sc.add_operator(Record::<f64>::new(), hsum);
            recs.push(hrec);
        }

        sc.run().await;

        for (i, hrec) in recs.iter().enumerate() {
            let series = sc.value(*hrec).expect("record exists");
            assert_eq!(series.timestamps(), tss(&[1, 2, 3]).as_slice());
            let expected = [100.0 + i as f64, 200.0 + i as f64, 300.0 + i as f64];
            assert_eq!(series.values(), &expected[..]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_sources_add_record() {
        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[1, 3]), vec![10.0_f64, 30.0]),
            Array::scalar(0.0_f64),
        ));
        let hb = sc.add_source(ArraySource::new(
            Series::from_vec(&[], tss(&[2, 3]), vec![20.0_f64, 40.0]),
            Array::scalar(0.0_f64),
        ));
        let hsum = sc.add_operator(Add::<f64>::new(), (ha, hb));
        let hrec = sc.add_operator(Record::<f64>::new(), hsum);

        sc.run().await;

        let series = sc.value(hrec).expect("record output exists");
        // ts=1: A=10,B=0 (init) → 10.  ts=2: A=10,B=20 → 30.  ts=3: A=30,B=40 → 70.
        assert_eq!(series.timestamps(), tss(&[1, 2, 3]).as_slice());
        assert_eq!(series.values(), &[10.0, 30.0, 70.0]);
    }
}

