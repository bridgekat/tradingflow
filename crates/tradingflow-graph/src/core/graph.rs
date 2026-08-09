use std::any::TypeId;
use std::ops::Range;
use std::sync::atomic::{self, AtomicUsize, Ordering};

use super::cell::ErasedCell;
use super::error::Error;
use super::node::{ComputeFn, Node, ResetFn};

/// Adjacency matrix stored in compressed sparse row format.
pub struct Adjacency {
    columns: Vec<usize>,
    row_ends: Vec<usize>,
}

impl Adjacency {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            row_ends: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.row_ends.is_empty()
    }

    pub fn len(&self) -> usize {
        self.row_ends.len()
    }

    pub fn get(&self, i: usize) -> &[usize] {
        let start = if i == 0 { 0 } else { self.row_ends[i - 1] };
        let end = self.row_ends[i];
        &self.columns[start..end]
    }

    pub fn push(&mut self, row: &[usize]) {
        self.columns.extend_from_slice(row);
        self.row_ends.push(self.columns.len());
    }

    pub fn transpose(&self, columns: usize) -> Self {
        assert!(self.columns.iter().all(|&c| c < columns));
        let mut transposed = Self::new();
        let mut counts = vec![0; columns];
        for i in 0..self.len() {
            for &j in self.get(i) {
                counts[j] += 1;
            }
        }
        let mut offsets = Vec::new();
        for count in &counts {
            offsets.push(transposed.columns.len());
            transposed.row_ends.push(transposed.columns.len() + *count);
            transposed.columns.extend(std::iter::repeat_n(0, *count));
        }
        for i in 0..self.len() {
            for &j in self.get(i) {
                transposed.columns[offsets[j]] = i;
                offsets[j] += 1;
            }
        }
        transposed
    }
}

impl Default for Adjacency {
    fn default() -> Self {
        Self::new()
    }
}

/// A pointer into an output slot inside the state of a [`Node`].
#[repr(transparent)]
struct SlotPtr(*const ());

// # Safety
//
// Synchronization of output slot accesses is explicitly handled:
//
// - Every slot has only one writer node (the node state owner).
// - Every slot holds a value of `T: Sync` per [`Node::new`] contract,
//   so concurrent reader nodes may share `&T`.
// - When a node runs, every successor node has `counter > 0` and is therefore
//   not running: concurrent readers never race a writer. This memory ordering
//   is enforced by Release and Acquire fences around the `counter` updates.
unsafe impl Send for SlotPtr {}
unsafe impl Sync for SlotPtr {}

/// A [`Node`] that is already fixed into a graph.
struct FixedNode {
    compute_fn: ComputeFn,
    reset_fn: ResetFn,
    input_range: Range<usize>,
    output_range: Range<usize>,
    state: ErasedCell,
    is_heavy: bool,
}

/// Builder for the core layer [`Graph`].
pub struct Builder {
    input_ptrs: Vec<SlotPtr>,
    output_ptrs: Vec<SlotPtr>,
    output_type_ids: Vec<TypeId>,
    input_from_outputs: Adjacency,
    nodes: Vec<FixedNode>,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            input_ptrs: Vec::new(),
            output_ptrs: Vec::new(),
            output_type_ids: Vec::new(),
            input_from_outputs: Adjacency::new(),
            nodes: Vec::new(),
        }
    }

    pub fn slot_type_id(&self, index: usize) -> TypeId {
        self.output_type_ids[index]
    }

    pub fn slot_ptr(&self, index: usize) -> *const () {
        self.output_ptrs[index].0
    }

    pub fn push(
        &mut self,
        op: Node,
        input_indices: &[usize],
    ) -> Result<(usize, Range<usize>), Error> {
        let (input_types, output_types, compute_fn, reset_fn, state, output_ptrs, is_heavy) =
            op.into_parts();

        let input_arity = input_types.len();
        if input_arity != input_indices.len() {
            return Err(Error::InputArity {
                expected: input_arity,
                actual: input_indices.len(),
            });
        }
        for (place, (&slot, &ty)) in input_indices.iter().zip(input_types.iter()).enumerate() {
            let num_slots = self.output_type_ids.len();
            if slot >= num_slots {
                return Err(Error::InputOutOfBounds {
                    place,
                    slot,
                    num_slots,
                });
            }
            if ty != self.output_type_ids[slot] {
                return Err(Error::InputType {
                    place,
                    slot,
                    expected: ty,
                    actual: self.output_type_ids[slot],
                });
            }
        }
        let output_arity = output_types.len();
        if output_arity != output_ptrs.len() {
            return Err(Error::OutputArity {
                expected: output_arity,
                actual: output_ptrs.len(),
            });
        }
        self.output_type_ids.extend(output_types);

        let input_begin = self.input_ptrs.len();
        let output_begin = self.output_ptrs.len();
        for &s in input_indices {
            self.input_ptrs.push(SlotPtr(self.slot_ptr(s)));
            self.input_from_outputs.push(&[s]);
        }
        self.output_ptrs
            .extend(output_ptrs.iter().map(|&p| SlotPtr(p)));
        let input_range = input_begin..self.input_ptrs.len();
        let output_range = output_begin..self.output_ptrs.len();

        let node_index = self.nodes.len();
        self.nodes.push(FixedNode {
            compute_fn,
            reset_fn,
            input_range,
            output_range: output_range.clone(),
            state,
            is_heavy,
        });
        Ok((node_index, output_range))
    }

    pub fn build(self) -> Graph {
        let Builder {
            input_ptrs,
            output_ptrs,
            output_type_ids,
            input_from_outputs,
            nodes,
        } = self;

        let input_ptrs = input_ptrs.into_boxed_slice();
        let output_ptrs = output_ptrs.into_boxed_slice();
        let output_type_ids = output_type_ids.into_boxed_slice();
        let nodes = nodes.into_boxed_slice();

        let mut input_owners = vec![usize::MAX; input_ptrs.len()].into_boxed_slice();
        for (i, node) in nodes.iter().enumerate() {
            for s in node.input_range.clone() {
                assert!(input_owners[s] == usize::MAX);
                input_owners[s] = i;
            }
        }

        let output_to_inputs = input_from_outputs.transpose(output_ptrs.len());
        let mut node_to_nodes = Adjacency::new();

        for (i, node) in nodes.iter().enumerate() {
            let mut to_nodes = Vec::new();
            for s in node.output_range.clone() {
                for &t in output_to_inputs.get(s) {
                    let j = input_owners[t];
                    assert!(i < j);
                    to_nodes.push(j);
                }
            }
            to_nodes.sort_unstable();
            to_nodes.dedup();
            node_to_nodes.push(&to_nodes);
        }

        let counters = (0..nodes.len()).map(|_| AtomicUsize::new(0)).collect();
        let is_dirty = (0..nodes.len()).map(|_| false).collect();

        Graph {
            input_ptrs,
            output_ptrs,
            output_type_ids,
            node_to_nodes,
            nodes,
            counters,
            roots: Vec::new(),
            dirty: Vec::new(),
            is_dirty,
            poisoned: false,
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

/// The type-erased core of a graph.
pub struct Graph {
    input_ptrs: Box<[SlotPtr]>,
    output_ptrs: Box<[SlotPtr]>,
    output_type_ids: Box<[TypeId]>,
    node_to_nodes: Adjacency,
    nodes: Box<[FixedNode]>,
    counters: Box<[AtomicUsize]>,
    roots: Vec<usize>,
    dirty: Vec<usize>,
    is_dirty: Box<[bool]>,
    poisoned: bool,
}

impl Graph {
    pub fn slot_type_id(&self, index: usize) -> TypeId {
        self.output_type_ids[index]
    }

    pub fn slot_ptr(&self, index: usize) -> *const () {
        assert!(!self.poisoned, "cannot access poisoned graph.");
        assert!(self.dirty.is_empty(), "cannot read unstabilized graph.");
        self.output_ptrs[index].0
    }

    pub fn state_mut(&mut self, index: usize) -> &mut ErasedCell {
        assert!(!self.poisoned, "cannot access poisoned graph.");
        assert!(
            self.nodes[index].input_range.is_empty(),
            "cannot mutate non-source node."
        );
        if !self.is_dirty[index] {
            self.is_dirty[index] = true;
            self.dirty.push(index);
        }
        &mut self.nodes[index].state
    }

    #[inline(always)]
    fn run_compute(&self, i: usize, context: &impl Sync) {
        let node = &self.nodes[i];
        let input_ptrs = slot_slice(&self.input_ptrs[node.input_range.clone()]);
        let state = node.state.get();
        let context = context as *const _ as *const ();
        unsafe { (node.compute_fn)(input_ptrs, state, context) };
    }

    #[inline(always)]
    fn run_reset(&self, i: usize, context: &impl Sync) {
        let node = &self.nodes[i];
        let input_ptrs = slot_slice(&self.input_ptrs[node.input_range.clone()]);
        let state = node.state.get();
        let context = context as *const _ as *const ();
        unsafe { (node.reset_fn)(input_ptrs, state, context) };
    }

    pub fn stabilize(&mut self, pool: &mut crate::pool::Pool, context: &impl Sync) {
        assert!(!self.poisoned, "cannot access poisoned graph.");
        self.poisoned = true;

        // Record the root nodes. The only source of dirty nodes, `state_mut`,
        // can be applied only on source nodes (nodes with no predecessors).
        std::mem::swap(&mut self.roots, &mut self.dirty);

        // Use pool for parallelism only if it has worker threads.
        let is_parallel = pool.num_other_threads() > 0;

        // Discover the dirty cone using depth-first search, accumulating in
        // `counter` each cone node's number of dirty predecessors. After this,
        // `self.dirty` stores all cone nodes in exit order (a reverse
        // topological order).
        fn dfs(
            i: usize,
            node_to_nodes: &Adjacency,
            is_dirty: &mut [bool],
            dirty: &mut Vec<usize>,
            counters: &mut [AtomicUsize],
            is_parallel: bool,
        ) {
            for &j in node_to_nodes.get(i) {
                if is_parallel {
                    *counters[j].get_mut() += 1;
                }
                if !is_dirty[j] {
                    is_dirty[j] = true;
                    dfs(j, node_to_nodes, is_dirty, dirty, counters, is_parallel);
                }
            }
            dirty.push(i);
        }
        for &i in self.roots.iter() {
            dfs(
                i,
                &self.node_to_nodes,
                &mut self.is_dirty,
                &mut self.dirty,
                &mut self.counters,
                is_parallel,
            );
        }

        // Call compute functions for nodes in the dirty cone,
        // in topological order.
        if is_parallel {
            pool.run(
                |scope| {
                    for &i in self.roots.iter() {
                        scope.spawn(i, self.nodes[i].is_heavy);
                    }
                },
                |i, scope| {
                    self.run_compute(i, context);
                    atomic::fence(Ordering::Release);
                    for &j in self.node_to_nodes.get(i) {
                        // Decrement the counter of each successor node.
                        // Upon reaching 0, all dirty predecessors have
                        // finished and the successor is ready to run.
                        // `Release` above and `Acquire` below ensure that the
                        // successor sees all writes to the dirty predecessors'
                        // outputs, via raw pointers.
                        if self.counters[j].fetch_sub(1, Ordering::Relaxed) == 1 {
                            atomic::fence(Ordering::Acquire);
                            scope.spawn(j, self.nodes[j].is_heavy);
                        }
                    }
                },
            );
        } else {
            for i in self.dirty.iter().rev().copied() {
                self.run_compute(i, context);
            }
        }
        self.roots.clear();

        // Call reset functions and reset cone membership for the next round,
        // in topological order.
        for i in self.dirty.iter().rev().copied() {
            self.run_reset(i, context);
            self.is_dirty[i] = false;
        }
        self.dirty.clear();

        // Clear poison flag on clean exit.
        self.poisoned = false;
    }
}

/// Views a slice of slot pointers as a raw slice of `*const ()`.
/// `SlotPtr` is `repr(transparent)` over `*const ()`, so the layouts agree.
fn slot_slice(slots: &[SlotPtr]) -> *const [*const ()] {
    std::ptr::slice_from_raw_parts(slots.as_ptr() as *const *const (), slots.len())
}
