//! `WavefrontScenario` — the public API for the wavefront runtime.

use std::any::TypeId;

use crate::data::{FlatRead, FlatWrite, InputTypes, Instant};
use crate::operator::ComputeFn;
use crate::scenario::handle::{Handle, InputTypesHandles};

use super::graph::{WavefrontGraph, WavefrontNode};
use super::operator::Operator as WfOperator;
use super::scheduler;
use super::source::Source;
use super::storage::VersionedRing;

/// Wavefront computation graph.
pub struct WavefrontScenario {
    graph: WavefrontGraph,
    timestamps: Vec<Instant>,
    node_meta: Vec<NodeMeta>,
}

struct NodeMeta {
    type_id: TypeId,
    #[allow(dead_code)]
    clone_fn: unsafe fn(*const u8) -> *mut u8,
}

impl WavefrontScenario {
    pub fn new() -> Self {
        Self {
            graph: WavefrontGraph::new(),
            timestamps: Vec::new(),
            node_meta: Vec::new(),
        }
    }

    /// Register a batch source.
    pub fn add_source<S: Source>(&mut self, source: S) -> Handle<S::Output> {
        let events = source.events();

        // Collect timestamps.
        for &(ts, _) in &events {
            if self.timestamps.is_empty() || ts > *self.timestamps.last().unwrap() {
                self.timestamps.push(ts);
            }
        }

        let init_output = source.init_output();
        let output_ptr = Box::into_raw(Box::new(init_output)) as *mut u8;

        let type_id = TypeId::of::<S::Output>();
        let output_drop = erased_drop_fn::<S::Output>;
        let output_clone = erased_clone_fn::<S::Output>;

        // Pre-fill versioned ring for all ticks.
        let mut versioned = VersionedRing::new();
        for (tick, &(_ts, ref value)) in events.iter().enumerate() {
            let ptr = unsafe { output_clone(output_ptr) };
            let output_ref = unsafe { &mut *(ptr as *mut S::Output) };
            S::write(value, output_ref);
            versioned.push(tick, ptr, output_drop);
        }

        let current_output = unsafe { output_clone(output_ptr) };

        let node = WavefrontNode {
            index: self.graph.nodes.len(),
            versioned,
            init_state_ptr: std::ptr::null_mut(),
            init_output_ptr: output_ptr,
            current_state_ptr: std::ptr::null_mut(),
            current_output_ptr: current_output,
            state_clone_fn: dummy_clone_fn,
            state_drop_fn: dummy_drop_fn,
            output_clone_fn: output_clone,
            output_drop_fn: output_drop,
            type_id,
            is_source: true,
            compute_fn: dummy_compute_fn,
            input_indices: Box::new([]),
            trigger_edges: Vec::new(),
            is_stateful: false,
        };

        let idx = self.graph.add_node(node);
        self.graph.source_indices.push(idx);
        self.node_meta.push(NodeMeta {
            type_id,
            clone_fn: output_clone,
        });

        Handle::new(idx)
    }

    /// Register an operator.
    pub fn add_operator<O: WfOperator>(
        &mut self,
        operator: O,
        inputs: impl Into<<O::Inputs as InputTypesHandles>::Handles>,
    ) -> Handle<O::Output>
    where
        O::Inputs: InputTypesHandles,
    {
        let handles = inputs.into();
        let arity = <O::Inputs as InputTypesHandles>::arity(&handles);
        let mut input_indices: Vec<usize> = vec![0; arity];
        {
            let mut writer = FlatWrite::new(&mut input_indices);
            <O::Inputs as InputTypesHandles>::write_node_indices(&handles, &mut writer);
        }

        // Collect input pointers from current graph state.
        let input_ptrs: Box<[*const u8]> = input_indices
            .iter()
            .map(|&idx| self.graph.nodes[idx].init_output_ptr as *const u8)
            .collect();

        // Init operator.
        let timestamp = self.timestamps.first().copied().unwrap_or(Instant::MIN);
        let mut ptr_reader = FlatRead::<*const u8>::new(&input_ptrs);
        let inputs_refs = unsafe { O::Inputs::refs_from_flat(&mut ptr_reader) };
        let (state, output) = operator.init(inputs_refs, timestamp);

        let output_ptr = Box::into_raw(Box::new(output)) as *mut u8;
        let state_ptr = Box::into_raw(Box::new(state)) as *mut u8;

        let type_id = TypeId::of::<O::Output>();
        let output_drop = erased_drop_fn::<O::Output>;
        let output_clone = erased_clone_fn::<O::Output>;
        let state_drop = erased_drop_fn::<O::State>;
        let state_clone = erased_clone_fn::<O::State>;

        let is_stateful = O::is_stateful();

        let compute_fn = make_erased_compute_fn::<O>();

        // current = clone of init (they start equal; stateful nodes diverge per tick)
        let current_state = unsafe { state_clone(state_ptr) };
        let current_output = unsafe { output_clone(output_ptr) };

        let node = WavefrontNode {
            index: self.graph.nodes.len(),
            versioned: VersionedRing::new(),
            init_state_ptr: state_ptr,
            init_output_ptr: output_ptr,
            current_state_ptr: current_state,
            current_output_ptr: current_output,
            state_clone_fn: state_clone,
            state_drop_fn: state_drop,
            output_clone_fn: output_clone,
            output_drop_fn: output_drop,
            type_id,
            is_source: false,
            compute_fn,
            input_indices: input_indices.clone().into(),
            trigger_edges: Vec::new(),
            is_stateful,
        };

        let output_idx = self.graph.add_node(node);
        for (pos, &input_idx) in input_indices.iter().enumerate() {
            self.graph.add_trigger_edge(input_idx, output_idx, pos);
        }

        self.node_meta.push(NodeMeta {
            type_id,
            clone_fn: output_clone,
        });

        Handle::new(output_idx)
    }

    /// Run the wavefront over all timestamps.
    pub fn run(&mut self) {
        scheduler::run_wavefront(&mut self.graph, &self.timestamps);
    }

    /// Read a node's output at a specific tick.
    pub fn value_at<T: Send + 'static>(&self, h: Handle<T>, tick: usize) -> &T {
        assert_eq!(
            self.node_meta[h.index()].type_id,
            TypeId::of::<T>(),
            "type mismatch at node {}",
            h.index(),
        );
        let node = &self.graph.nodes[h.index()];
        let ptr = node
            .versioned
            .get(tick)
            .unwrap_or_else(|| panic!("tick {tick} not computed for node {}", h.index()));
        unsafe { &*(ptr as *const T) }
    }

    /// Number of timestamps.
    pub fn num_ticks(&self) -> usize {
        self.timestamps.len()
    }
}

impl Default for WavefrontScenario {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------
// Erased function pointers
// -----------------------------------------------------------------------

unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}

unsafe fn erased_clone_fn<T: Clone>(ptr: *const u8) -> *mut u8 {
    let val = unsafe { &*(ptr as *const T) };
    Box::into_raw(Box::new(val.clone())) as *mut u8
}

fn dummy_compute_fn(
    _state: *mut u8,
    _input_ptrs: &[*const u8],
    _output: *mut u8,
    _timestamp: Instant,
    _produced_words: &[u64],
    _produced_bit_off: usize,
    _produced_num_inputs: usize,
) -> bool {
    false
}

unsafe fn dummy_clone_fn(_ptr: *const u8) -> *mut u8 {
    std::ptr::null_mut()
}

unsafe fn dummy_drop_fn(_ptr: *mut u8) {}

/// Build a type-erased compute function for an [`WfOperator`].
fn make_erased_compute_fn<O: WfOperator>() -> ComputeFn {
    unsafe fn erased<O: WfOperator>(
        state_ptr: *mut u8,
        input_ptrs: &[*const u8],
        output_ptr: *mut u8,
        timestamp: Instant,
        produced_words: &[u64],
        produced_bit_off: usize,
        produced_num_inputs: usize,
    ) -> bool {
        let state = unsafe { &mut *(state_ptr as *mut O::State) };
        let mut ptr_reader = FlatRead::new(input_ptrs);
        let inputs = unsafe { O::Inputs::refs_from_flat(&mut ptr_reader) };
        let mut bit_reader =
            crate::BitRead::from_parts(produced_words, produced_bit_off, produced_num_inputs);
        let produced = O::Inputs::produced_from_flat(&mut bit_reader);
        let output = unsafe { &mut *(output_ptr as *mut O::Output) };
        O::compute(state, inputs, output, timestamp, produced)
    }
    erased::<O>
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;
    use crate::experimental::operators::{Add, Diff, Multiply};

    #[allow(dead_code)]
    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    fn tss(xs: &[i64]) -> Vec<Instant> {
        xs.iter().copied().map(Instant::from_nanos).collect()
    }

    fn array_source(
        timestamps: Vec<Instant>,
        values: &[f64],
        shape: &[usize],
    ) -> crate::experimental::source::ArraySource<f64> {
        crate::experimental::source::ArraySource::new(timestamps, values, shape, Array::zeros(shape))
    }

    #[test]
    fn test_single_stateless() {
        let mut sc = WavefrontScenario::new();
        let h_a = sc.add_source(array_source(tss(&[1, 2, 3]), &[10.0, 20.0, 30.0], &[]));
        let h_out = sc.add_operator(Add::new(), (h_a, h_a));

        sc.run();

        assert_eq!(sc.value_at::<Array<f64>>(h_out, 0).as_slice(), &[20.0]);
        assert_eq!(sc.value_at::<Array<f64>>(h_out, 1).as_slice(), &[40.0]);
        assert_eq!(sc.value_at::<Array<f64>>(h_out, 2).as_slice(), &[60.0]);
    }

    #[test]
    fn test_single_stateful() {
        let mut sc = WavefrontScenario::new();
        let h_a = sc.add_source(array_source(
            tss(&[1, 2, 3, 4]),
            &[10.0, 25.0, 20.0, 30.0],
            &[],
        ));
        let h_out = sc.add_operator(Diff::new(), h_a);

        sc.run();

        assert!(sc.value_at::<Array<f64>>(h_out, 0).as_slice()[0].is_nan());
        assert_eq!(sc.value_at::<Array<f64>>(h_out, 1).as_slice(), &[15.0]);
        assert_eq!(sc.value_at::<Array<f64>>(h_out, 2).as_slice(), &[-5.0]);
        assert_eq!(sc.value_at::<Array<f64>>(h_out, 3).as_slice(), &[10.0]);
    }

    #[test]
    fn test_stateless_chain() {
        let mut sc = WavefrontScenario::new();
        let h_a = sc.add_source(array_source(tss(&[1, 2, 3]), &[1.0, 2.0, 3.0], &[]));
        let h_b = sc.add_source(array_source(tss(&[1, 2, 3]), &[10.0, 20.0, 30.0], &[]));
        let h_add = sc.add_operator(Add::new(), (h_a, h_b));
        let h_mul = sc.add_operator(Multiply::new(), (h_add, h_add));

        sc.run();

        assert_eq!(sc.value_at::<Array<f64>>(h_mul, 0).as_slice(), &[121.0]);
        assert_eq!(sc.value_at::<Array<f64>>(h_mul, 1).as_slice(), &[484.0]);
        assert_eq!(sc.value_at::<Array<f64>>(h_mul, 2).as_slice(), &[1089.0]);
    }

    #[test]
    fn test_stateless_then_stateful() {
        let mut sc = WavefrontScenario::new();
        let h_a = sc.add_source(array_source(
            tss(&[1, 2, 3, 4]),
            &[5.0, 12.5, 10.0, 15.0],
            &[],
        ));
        let h_double = sc.add_operator(Add::new(), (h_a, h_a));
        let h_diff = sc.add_operator(Diff::new(), h_double);

        sc.run();

        assert!(sc.value_at::<Array<f64>>(h_diff, 0).as_slice()[0].is_nan());
        assert_eq!(sc.value_at::<Array<f64>>(h_diff, 1).as_slice(), &[15.0]);
        assert_eq!(sc.value_at::<Array<f64>>(h_diff, 2).as_slice(), &[-5.0]);
        assert_eq!(sc.value_at::<Array<f64>>(h_diff, 3).as_slice(), &[10.0]);
    }

    #[test]
    fn test_two_branches() {
        let mut sc = WavefrontScenario::new();
        let h_a = sc.add_source(array_source(
            tss(&[1, 2, 3, 4]),
            &[10.0, 25.0, 20.0, 30.0],
            &[],
        ));
        let h_diff1 = sc.add_operator(Diff::new(), h_a);
        let h_diff2 = sc.add_operator(Diff::new(), h_a);

        sc.run();

        for t in 1..4 {
            assert_eq!(
                sc.value_at::<Array<f64>>(h_diff1, t).as_slice(),
                sc.value_at::<Array<f64>>(h_diff2, t).as_slice(),
            );
        }
    }
}
