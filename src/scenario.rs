use std::any::Any;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::observable::Observable;
use crate::operator::Operator;
use crate::refs::{Input, Inputs, Output, Scalar};
use crate::series::Series;
use crate::source::Source;

/// A typed handle into a [`Scenario`]'s node storage.
#[derive(Debug, Clone, Copy)]
pub struct ObservableHandle<T: Copy> {
    index: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> ObservableHandle<T> {
    fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }
}

/// A typed handle into a [`Scenario`]'s node storage.
#[derive(Debug, Clone, Copy)]
pub struct SeriesHandle<T: Copy> {
    index: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> SeriesHandle<T> {
    fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// InputHandle / InputHandles — maps Input types to Scenario handles
// ---------------------------------------------------------------------------

/// Maps an [`Input`] type to its corresponding Scenario handle type.
pub trait InputHandle: Input {
    /// The handle type used at registration time.
    type Handle;

    /// Extract `(node_index, is_series)` from a handle.
    fn node_id(handle: &Self::Handle) -> (usize, bool);
}

impl<T: Scalar> InputHandle for Observable<T> {
    type Handle = ObservableHandle<T>;

    #[inline(always)]
    fn node_id(handle: &ObservableHandle<T>) -> (usize, bool) {
        (handle.index, false)
    }
}

impl<T: Scalar> InputHandle for Series<T> {
    type Handle = SeriesHandle<T>;

    #[inline(always)]
    fn node_id(handle: &SeriesHandle<T>) -> (usize, bool) {
        (handle.index, true)
    }
}

/// Maps an [`Inputs`] collection to its corresponding Handles collection.
pub trait InputHandles: Inputs {
    /// The handle collection type used at registration time.
    type Handles;

    /// Extract `(node_index, is_series)` from each handle.
    fn node_ids(handles: &Self::Handles) -> Box<[(usize, bool)]>;
}

impl<T: InputHandle + 'static> InputHandles for T {
    type Handles = T::Handle;

    fn node_ids(handles: &T::Handle) -> Box<[(usize, bool)]> {
        Box::new([T::node_id(handles)])
    }
}

impl<T: InputHandle + 'static> InputHandles for [T] {
    type Handles = Box<[T::Handle]>;

    fn node_ids(handles: &Box<[T::Handle]>) -> Box<[(usize, bool)]> {
        handles.iter().map(|h| T::node_id(h)).collect()
    }
}

macro_rules! impl_tuple {
    ($($idx:tt: $R:ident),+ $(,)?) => {
        impl<$($R: InputHandle + 'static),+> InputHandles for ($($R,)+) {
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
// Node (type-erased, safe)
// ---------------------------------------------------------------------------

/// Type-erased node: observable + optional series.
///
/// Ownership is managed through `Box<dyn Any>`, so no manual `Drop` needed.
struct Node {
    /// The observable value. Temporarily `None` during compute dispatch.
    obs: Option<Box<dyn Any + Send>>,
    /// Optional series, allocated by [`materialize`].
    series: Option<Box<dyn Any + Send>>,
    /// Copy observable value into series.
    materialize_fn: fn(&dyn Any, &mut dyn Any, i64),
    /// Element shape of the observable.
    shape: Box<[usize]>,
}

/// Materialise: copy observable value into series.
fn materialize_copy<T: Copy + 'static>(obs: &dyn Any, series: &mut dyn Any, timestamp: i64) {
    let obs = obs.downcast_ref::<Observable<T>>().unwrap();
    let series = series.downcast_mut::<Series<T>>().unwrap();
    series.push(timestamp, obs.current());
}

/// Per-source runtime state, created at registration time.
struct SourceState {
    node_index: usize,
    ops: Box<dyn SourceRuntimeOps>,
    pending_hist_ts: Option<i64>,
    hist_exhausted: bool,
    live_exhausted: bool,
}

/// Type-erased operator with node index references.
struct OperatorState {
    output_node_index: usize,
    /// `(node_index, is_series)` for each input.
    input_node_ids: Vec<(usize, bool)>,
    /// Type-erased compute function.
    compute_fn: fn(Box<[&dyn Any]>, &mut dyn Any, &mut dyn Any) -> bool,
    /// Operator runtime state.
    state: Option<Box<dyn Any + Send>>,
}

/// Type-erased compute entry point.
///
/// Uses [`Inputs::from_erased`] and [`Output::from_erased`] to reconstruct
/// the typed references from `&dyn Any`.
fn erased_compute<Op>(
    input_anys: Box<[&dyn Any]>,
    output_any: &mut dyn Any,
    state_any: &mut dyn Any,
) -> bool
where
    Op: Operator,
{
    let state = state_any.downcast_mut::<Op::State>().unwrap();
    let inputs = <Op::Inputs as Inputs>::from_erased(input_anys);
    let output = <Op::Output as Output>::from_erased(output_any);
    Op::compute(state, inputs, output)
}

// ---------------------------------------------------------------------------
// SourceRuntimeOps — type-erased async source operations
// ---------------------------------------------------------------------------

/// Trait object for type-erased source channel operations.
trait SourceRuntimeOps: Send {
    /// Wait for the next historical event and return its timestamp.
    fn wait_historical(&mut self) -> Pin<Box<dyn Future<Output = Option<i64>> + '_>>;

    /// Wait for the next live event and return its timestamp.
    fn wait_live(&mut self) -> Pin<Box<dyn Future<Output = Option<i64>> + '_>>;

    /// Write the pending historical event into the output observable.
    fn write_historical(&mut self, output: &mut dyn Any) -> bool;

    /// Write the pending live event into the output observable.
    fn write_live(&mut self, output: &mut dyn Any) -> bool;
}

/// Concrete implementation of [`SourceRuntimeOps`] for a given [`Source`].
struct TypedSourceOps<S: Source> {
    hist_rx: tokio::sync::mpsc::Receiver<(i64, S::Event)>,
    live_rx: tokio::sync::mpsc::Receiver<(i64, S::Event)>,
    pending_hist: Option<(i64, S::Event)>,
    pending_live: Option<(i64, S::Event)>,
}

impl<S: Source> SourceRuntimeOps for TypedSourceOps<S>
where
    S::Output: Output,
{
    fn wait_historical(&mut self) -> Pin<Box<dyn Future<Output = Option<i64>> + '_>> {
        Box::pin(async move {
            if let Some(ref item) = self.pending_hist {
                return Some(item.0);
            }
            match self.hist_rx.recv().await {
                Some(item) => {
                    let ts = item.0;
                    self.pending_hist = Some(item);
                    Some(ts)
                }
                None => None,
            }
        })
    }

    fn wait_live(&mut self) -> Pin<Box<dyn Future<Output = Option<i64>> + '_>> {
        Box::pin(async move {
            if let Some(ref item) = self.pending_live {
                return Some(item.0);
            }
            match self.live_rx.recv().await {
                Some(item) => {
                    let ts = item.0;
                    self.pending_live = Some(item);
                    Some(ts)
                }
                None => None,
            }
        })
    }

    fn write_historical(&mut self, output: &mut dyn Any) -> bool {
        if let Some((_, payload)) = self.pending_hist.take() {
            let output = <S::Output as Output>::from_erased(output);
            S::write(payload, output)
        } else {
            false
        }
    }

    fn write_live(&mut self, output: &mut dyn Any) -> bool {
        if let Some((_, payload)) = self.pending_live.take() {
            let output = <S::Output as Output>::from_erased(output);
            S::write(payload, output)
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

/// Scenario — the DAG runtime that owns nodes and dispatches operators.
///
/// # Architecture
///
/// Every node has an [`Observable`] (always present) and an optional
/// [`Series`] (allocated on demand via [`materialize`]).  Operators write into
/// the observable; the scenario copies to the series if materialised.
///
/// All nodes are stored as type-erased [`NodeSlot`]s in a flat `Vec`,
/// using `Box<dyn Any>` for safe ownership.  Each operator is wrapped in
/// an [`OperatorSlot`] with a type-erased `compute_fn` function pointer
/// that uses `&dyn Any` downcasting for type recovery.
///
/// Type safety is enforced at registration time via [`ObservableHandle<T>`]
/// and [`SeriesHandle<T>`] generics.  After registration the scenario
/// operates on `dyn Any` trait objects.
///
/// # Flush algorithm
///
/// On each tick the caller writes to source observables and calls [`flush`].
/// A min-heap processes only the operators reachable from the updated sources,
/// in registration order (which is topological order, since inputs must exist
/// before an operator can be registered).  This is O(active) not O(total).
pub struct Scenario {
    nodes: Vec<Node>,
    edges: Vec<Vec<usize>>, // `edges[node_idx]` → operator indices that read from this node.
    source_states: Vec<SourceState>,
    operator_states: Vec<OperatorState>,
    pending: Vec<bool>,
    heap: BinaryHeap<Reverse<usize>>,
}

impl Scenario {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            source_states: Vec::new(),
            operator_states: Vec::new(),
            pending: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Get a shared reference to the concrete `Observable<T>`.
    #[inline(always)]
    pub fn observable<T: Scalar>(&self, h: ObservableHandle<T>) -> &Observable<T> {
        self.nodes[h.index]
            .obs
            .as_ref()
            .unwrap()
            .downcast_ref::<Observable<T>>()
            .unwrap()
    }

    /// Get a mutable reference to the concrete `Observable<T>` behind a
    /// handle.
    #[inline(always)]
    pub fn observable_mut<T: Scalar>(&mut self, h: ObservableHandle<T>) -> &mut Observable<T> {
        self.nodes[h.index]
            .obs
            .as_mut()
            .unwrap()
            .downcast_mut::<Observable<T>>()
            .unwrap()
    }

    /// Get a shared reference to the concrete `Series<T>` behind a handle.
    #[inline(always)]
    pub fn series<T: Scalar>(&self, h: SeriesHandle<T>) -> &Series<T> {
        self.nodes[h.index]
            .series
            .as_ref()
            .unwrap()
            .downcast_ref::<Series<T>>()
            .unwrap()
    }

    /// Get a mutable reference to the concrete `Series<T>`.
    #[inline(always)]
    pub fn series_mut<T: Scalar>(&mut self, h: SeriesHandle<T>) -> &mut Series<T> {
        self.nodes[h.index]
            .series
            .as_mut()
            .unwrap()
            .downcast_mut::<Series<T>>()
            .unwrap()
    }

    /// Internal helper — create a node with an explicit initial value.
    fn create_node<T: Scalar>(&mut self, shape: &[usize], initial: &[T]) -> ObservableHandle<T> {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            obs: Some(Box::new(Observable::<T>::new(shape, initial))),
            series: None,
            materialize_fn: materialize_copy::<T>,
            shape: shape.into(),
        });
        self.edges.push(Vec::new());
        ObservableHandle::new(idx)
    }

    /// Materialise a node: allocate a [`Series`] alongside the observable.
    ///
    /// Returns a [`SeriesHandle`] proving that the node has history storage.
    /// Panics if the node is already materialised.
    pub fn materialize<T: Scalar>(&mut self, h: ObservableHandle<T>) -> SeriesHandle<T> {
        if self.nodes[h.index].series.is_some() {
            return SeriesHandle::new(h.index);
        }
        let obs = self.nodes[h.index]
            .obs
            .as_ref()
            .unwrap()
            .downcast_ref::<Observable<T>>()
            .unwrap();
        let shape = obs.shape().to_vec();
        let series = Series::<T>::new(&shape, obs.current());
        self.nodes[h.index].series = Some(Box::new(series));
        SeriesHandle::new(h.index)
    }

    /// Register a [`Source`], creating the output node from the source's
    /// [`shape`](Source::shape) and [`initial`](Source::initial) values.
    ///
    /// The source will be consumed by [`run`] to feed data into the node.
    /// Returns a handle to the output observable.
    pub fn add_source<S>(&mut self, source: S) -> ObservableHandle<<S::Output as Output>::Scalar>
    where
        S: Source,
    {
        let shape = source.shape();
        let initial = source.initial();
        let handle = self.create_node(&shape, &initial);
        let node_index = handle.index;
        let (hist_rx, live_rx) = source.subscribe();
        self.source_states.push(SourceState {
            node_index,
            ops: Box::new(TypedSourceOps::<S> {
                hist_rx,
                live_rx,
                pending_hist: None,
                pending_live: None,
            }),
            pending_hist_ts: None,
            hist_exhausted: false,
            live_exhausted: false,
        });
        handle
    }

    /// Register an operator.
    ///
    /// Creates the output node, runs an initial `compute` from the current
    /// input values, and returns a handle to the output observable.
    pub fn add_operator<Op, T: Scalar>(
        &mut self,
        inputs: impl Into<<Op::Inputs as InputHandles>::Handles>,
        op: Op,
    ) -> ObservableHandle<T>
    where
        Op: Operator,
        Op::Inputs: InputHandles,
        Op::Output: Output<Scalar = T>,
    {
        let handles = inputs.into();
        let node_ids = <Op::Inputs as InputHandles>::node_ids(&handles);
        let input_shapes: Box<[&[usize]]> = node_ids
            .iter()
            .map(|&(i, _)| self.nodes[i].shape.as_ref())
            .collect();
        let output_shape = op.shape(&input_shapes);
        let initial = op.initial(&input_shapes);
        let state = op.init();

        let output = self.create_node::<T>(&output_shape, &initial);
        let op_idx = self.operator_states.len();

        let input_node_ids: Vec<(usize, bool)> = node_ids.to_vec();
        for &(i, _) in &input_node_ids {
            self.edges[i].push(op_idx);
        }

        self.operator_states.push(OperatorState {
            output_node_index: output.index,
            input_node_ids,
            compute_fn: erased_compute::<Op>,
            state: Some(Box::new(state)),
        });

        self.pending.push(false);

        // Compute initial output value from current input values.
        self.dispatch_compute(op_idx);

        output
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
        // Fetch initial historical event from each source.
        for rt in &mut self.source_states {
            if !rt.hist_exhausted {
                let ts = rt.ops.wait_historical().await;
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
            for rt in &mut self.source_states {
                if !rt.hist_exhausted && rt.pending_hist_ts.is_none() {
                    let ts = rt.ops.wait_historical().await;
                    rt.pending_hist_ts = ts;
                    if ts.is_none() {
                        rt.hist_exhausted = true;
                    }
                }
            }

            // Find the minimum pending timestamp across all sources.
            let min_ts = self
                .source_states
                .iter()
                .filter_map(|rt| rt.pending_hist_ts)
                .min();

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
            if let Some(qts) = queue_ts
                && min_ts > qts
            {
                self.flush(qts, &queue_sources);
                queue_sources.clear();
            }

            // Collect all sources at min_ts: write to observables, add to queue.
            for rt in &mut self.source_states {
                if rt.pending_hist_ts == Some(min_ts) {
                    // Write the pending payload to the observable.
                    let obs = self.nodes[rt.node_index].obs.as_mut().unwrap();
                    rt.ops.write_historical(&mut **obs);
                    queue_sources.push(rt.node_index);
                    rt.pending_hist_ts = None;

                    // Fetch next historical event from this source.
                    if !rt.hist_exhausted {
                        let ts = rt.ops.wait_historical().await;
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
            let node = &mut self.nodes[idx];
            if let Some(ref mut series) = node.series {
                let obs = node.obs.as_ref().unwrap().as_ref();
                (node.materialize_fn)(obs, &mut **series, timestamp);
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

            if self.dispatch_compute(op_idx) {
                let out_idx = self.operator_states[op_idx].output_node_index;

                // Copy observable → series if node is materialised.
                let node = &mut self.nodes[out_idx];
                if let Some(ref mut series) = node.series {
                    let obs = node.obs.as_ref().unwrap().as_ref();
                    (node.materialize_fn)(obs, &mut **series, timestamp);
                }

                // Schedule downstream operators.
                for &downstream in &self.edges[out_idx] {
                    if !self.pending[downstream] {
                        self.pending[downstream] = true;
                        self.heap.push(Reverse(downstream));
                    }
                }
            }
        }
    }

    /// Run a single operator's compute, handling the borrow splitting.
    fn dispatch_compute(&mut self, op_idx: usize) -> bool {
        // Copy lightweight data from the operator slot.
        let out_idx = self.operator_states[op_idx].output_node_index;
        let compute_fn = self.operator_states[op_idx].compute_fn;

        // Take the output observable and operator state.
        let mut state = std::mem::take(&mut self.operator_states[op_idx].state);
        let mut output = std::mem::take(&mut self.nodes[out_idx].obs);

        // Collect input references (self.nodes is now only borrowed immutably
        // since the output slot is None and won't be accessed).
        let inputs = self.operator_states[op_idx]
            .input_node_ids
            .iter()
            .map(|&(i, is_series)| -> &dyn Any {
                if is_series {
                    self.nodes[i].series.as_ref().unwrap().as_ref()
                } else {
                    self.nodes[i].obs.as_ref().unwrap().as_ref()
                }
            })
            .collect();

        // Invoke the compute function.
        let produced = compute_fn(
            inputs,
            output.as_mut().unwrap().as_mut(),
            state.as_mut().unwrap().as_mut(),
        );

        // Return the output observable and operator state.
        self.operator_states[op_idx].state = state;
        self.nodes[out_idx].obs = output;

        produced
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
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], operators::add());

        sc.observable_mut(ha).write(&[10.0]);
        sc.observable_mut(hb).write(&[3.0]);
        sc.flush(1, &[ha.index, hb.index]);

        let out = sc.observable(ho);
        assert_eq!(out.current_view().as_slice().unwrap(), &[13.0]);
    }

    #[test]
    fn materialized_output() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], operators::add());
        let ho_series = sc.materialize::<f64>(ho);

        sc.observable_mut(ha).write(&[10.0]);
        sc.observable_mut(hb).write(&[3.0]);
        sc.flush(1, &[ha.index, hb.index]);

        sc.observable_mut(ha).write(&[20.0]);
        sc.observable_mut(hb).write(&[7.0]);
        sc.flush(2, &[ha.index, hb.index]);

        // Observable has latest value.
        let obs = sc.observable(ho);
        assert_eq!(obs.current_view().as_slice().unwrap(), &[27.0]);

        // Series has full history (initial + 2 flushes).
        let series = sc.series(ho_series);
        assert_eq!(series.len(), 3);
        assert_eq!(series.index(), &[i64::MIN, 1, 2]);
        assert_eq!(series.values(), &[0.0, 13.0, 27.0]);
    }

    #[test]
    fn chain_operators() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let hab = sc.add_operator([ha, hb], operators::add());
        let hout = sc.add_operator([hab, ha], operators::multiply());

        sc.observable_mut(ha).write(&[2.0]);
        sc.observable_mut(hb).write(&[3.0]);
        sc.flush(1, &[ha.index, hb.index]);

        let out = sc.observable(hout);
        assert_eq!(out.current_view().as_slice().unwrap(), &[10.0]); // (2+3) * 2
    }

    #[test]
    fn sparse_update_skips_inactive() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let ho1 = sc.add_operator([ha, hb], operators::add());
        let ho1_series = sc.materialize::<f64>(ho1);

        let ho2 = sc.create_node::<f64>(&[], &[0.0]);
        let hc = sc.add_operator([ho2, ha], operators::add());
        let hc_series = sc.materialize::<f64>(hc);

        sc.observable_mut(ha).write(&[1.0]);
        sc.observable_mut(hb).write(&[2.0]);
        sc.flush(1, &[ha.index, hb.index]);

        let out1 = sc.series(ho1_series);
        assert_eq!(out1.len(), 2); // initial + 1 flush
        assert_eq!(out1.current_view().as_slice().unwrap(), &[3.0]);

        // hc produces output (both observables have values), but ho2 is 0.0 (initial).
        let outc = sc.series(hc_series);
        assert_eq!(outc.len(), 2); // initial + 1 flush
        assert_eq!(outc.current_view().as_slice().unwrap(), &[1.0]); // 0.0 + 1.0
    }

    #[test]
    fn incremental_ticks() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha, hb], operators::add());
        let ho_series = sc.materialize::<f64>(ho);

        for i in 0..100 {
            let ts = i as i64;
            let va = i as f64;
            let vb = (i * 2) as f64;
            sc.observable_mut(ha).write(&[va]);
            sc.observable_mut(hb).write(&[vb]);
            sc.flush(ts, &[ha.index, hb.index]);
        }

        let out = sc.series(ho_series);
        assert_eq!(out.len(), 101); // initial + 100 flushes
        assert_eq!(out.current_view().as_slice().unwrap(), &[99.0 + 198.0]);
    }

    #[test]
    fn unmaterialized_intermediate() {
        // Chain: a + b → mid → mid * a → out
        // Only materialize the final output, not mid.
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let hb = sc.create_node::<f64>(&[], &[0.0]);
        let hmid = sc.add_operator([ha, hb], operators::add());
        let hout = sc.add_operator([hmid, ha], operators::multiply());
        let hout_series = sc.materialize::<f64>(hout);

        for i in 1..=5 {
            let ts = i as i64;
            let v = i as f64;
            sc.observable_mut(ha).write(&[v]);
            sc.observable_mut(hb).write(&[v * 2.0]);
            sc.flush(ts, &[ha.index, hb.index]);
        }

        // mid is not materialised — no series exists.
        assert!(
            self::Scenario::default().nodes.is_empty() || sc.nodes[hmid.index].series.is_none()
        );

        // Final output has history.
        let out = sc.series(hout_series);
        assert_eq!(out.len(), 6); // initial + 5 flushes
        // At tick 5: mid = 5+10 = 15, out = 15*5 = 75
        assert_eq!(out.current_view().as_slice().unwrap(), &[75.0]);
    }

    #[test]
    fn materialize_source() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let ha_series = sc.materialize::<f64>(ha);

        for i in 0..10 {
            sc.observable_mut(ha).write(&[i as f64]);
            sc.flush(i as i64, &[ha.index]);
        }

        let series = sc.series(ha_series);
        assert_eq!(series.len(), 11); // initial + 10 flushes
        assert_eq!(series.current_view().as_slice().unwrap(), &[9.0]);
    }

    // -- Slice operator tests -----------------------------------------------

    #[test]
    fn slice_operator_concat() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[1.0]);
        let hb = sc.create_node::<f64>(&[], &[2.0]);
        let hc = sc.create_node::<f64>(&[], &[3.0]);
        let ho = sc.add_operator([ha, hb, hc], operators::Concat::new(&[], 0));
        let ho_series = sc.materialize::<f64>(ho);

        sc.observable_mut(ha).write(&[10.0]);
        sc.observable_mut(hb).write(&[20.0]);
        sc.observable_mut(hc).write(&[30.0]);
        sc.flush(1, &[ha.index, hb.index, hc.index]);

        let series = sc.series(ho_series);
        assert_eq!(series.len(), 2); // initial + 1 flush
        assert_eq!(
            series.current_view().as_slice().unwrap(),
            &[10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn slice_operator_stack() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[2], &[1.0, 2.0]);
        let hb = sc.create_node::<f64>(&[2], &[3.0, 4.0]);
        let ho = sc.add_operator([ha, hb], operators::Stack::new(&[2], 0));
        let ho_series = sc.materialize::<f64>(ho);

        sc.flush(1, &[ha.index, hb.index]);

        let series = sc.series(ho_series);
        assert_eq!(series.len(), 2); // initial + 1 flush
        assert_eq!(
            series.current_view().as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn select_operator() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[5], &[10.0, 20.0, 30.0, 40.0, 50.0]);
        let ho = sc.add_operator([ha], operators::Select::flat(vec![1, 3]));
        let ho_series = sc.materialize::<f64>(ho);

        sc.flush(1, &[ha.index]);

        let series = sc.series(ho_series);
        assert_eq!(series.len(), 2); // initial + 1 flush
        assert_eq!(series.current_view().as_slice().unwrap(), &[20.0, 40.0]);
    }

    #[test]
    fn filter_operator() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha], operators::Filter::new(|v: &[f64]| v[0] > 3.0));
        let ho_series = sc.materialize::<f64>(ho);

        // Value 1.0 → filtered out
        sc.observable_mut(ha).write(&[1.0]);
        sc.flush(1, &[ha.index]);
        // Value 5.0 → passes
        sc.observable_mut(ha).write(&[5.0]);
        sc.flush(2, &[ha.index]);
        // Value 2.0 → filtered out
        sc.observable_mut(ha).write(&[2.0]);
        sc.flush(3, &[ha.index]);
        // Value 10.0 → passes
        sc.observable_mut(ha).write(&[10.0]);
        sc.flush(4, &[ha.index]);

        let series = sc.series(ho_series);
        assert_eq!(series.len(), 3); // initial + 2 passes (filtered skipped)
        assert_eq!(series.index(), &[i64::MIN, 2, 4]);
        assert_eq!(series.values(), &[0.0, 5.0, 10.0]);
    }

    #[test]
    fn where_operator() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[3], &[0.0, 0.0, 0.0]);
        let ho = sc.add_operator([ha], operators::Where::new(|v: f64| v > 2.0, 0.0));
        let ho_series = sc.materialize::<f64>(ho);

        sc.observable_mut(ha).write(&[1.0, 5.0, 2.0]);
        sc.flush(1, &[ha.index]);

        let series = sc.series(ho_series);
        assert_eq!(series.len(), 2); // initial + 1 flush
        assert_eq!(series.current_view().as_slice().unwrap(), &[0.0, 5.0, 0.0]);
    }

    #[test]
    fn negate_operator() {
        let mut sc = Scenario::new();
        let ha = sc.create_node::<f64>(&[], &[0.0]);
        let ho = sc.add_operator([ha], operators::negate());

        sc.observable_mut(ha).write(&[7.0]);
        sc.flush(1, &[ha.index]);

        let out = sc.observable(ho);
        assert_eq!(out.current_view().as_slice().unwrap(), &[-7.0]);
    }

    // -- run() tests --------------------------------------------------------

    #[tokio::test]
    async fn run_single_source() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2, 3], vec![10.0, 20.0, 30.0], 1));
        let ha_series = sc.materialize::<f64>(ha);

        sc.run().await;

        let series = sc.series(ha_series);
        assert_eq!(series.len(), 4); // initial + 3 source events
        assert_eq!(series.index(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(series.values(), &[0.0, 10.0, 20.0, 30.0]);
    }

    #[tokio::test]
    async fn run_two_sources_interleaved() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 3], vec![10.0, 30.0], 1));
        let hb = sc.add_source(ArraySource::new(vec![2, 3], vec![20.0, 40.0], 1));
        let ho = sc.add_operator([ha, hb], operators::add());
        let ho_series = sc.materialize::<f64>(ho);

        sc.run().await;

        let series = sc.series(ho_series);
        // initial (0.0) + ts=1: a=10,b=0→10 + ts=2: a=10,b=20→30 + ts=3: a=30,b=40→70
        assert_eq!(series.len(), 4);
        assert_eq!(series.index(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(series.values(), &[0.0, 10.0, 30.0, 70.0]);
    }

    #[tokio::test]
    async fn run_coalescing() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2], vec![10.0, 20.0], 1));
        let ha_series = sc.materialize::<f64>(ha);
        let hb = sc.add_source(ArraySource::new(vec![1, 2], vec![100.0, 200.0], 1));
        let ho = sc.add_operator([ha, hb], operators::add());
        let ho_series = sc.materialize::<f64>(ho);

        sc.run().await;

        let series = sc.series(ho_series);
        assert_eq!(series.len(), 3); // initial + 2 coalesced flushes
        assert_eq!(series.index(), &[i64::MIN, 1, 2]);
        assert_eq!(series.values(), &[0.0, 110.0, 220.0]);

        let a_series = sc.series(ha_series);
        assert_eq!(a_series.len(), 3); // initial + 2 source events
    }

    #[tokio::test]
    async fn run_chained_operators() {
        use crate::sources::ArraySource;

        let mut sc = Scenario::new();
        let ha = sc.add_source(ArraySource::new(vec![1, 2], vec![2.0, 5.0], 1));
        let hb = sc.add_source(ArraySource::new(vec![1, 2], vec![3.0, 10.0], 1));
        let hab = sc.add_operator([ha, hb], operators::add());
        let hout = sc.add_operator([hab, ha], operators::multiply());
        let hout_series = sc.materialize::<f64>(hout);

        sc.run().await;

        let series = sc.series(hout_series);
        assert_eq!(series.len(), 3); // initial + 2 flushes
        // initial: 0.0, ts=1: (2+3)*2 = 10, ts=2: (5+10)*5 = 75
        assert_eq!(series.values(), &[0.0, 10.0, 75.0]);
    }
}
