use std::any::TypeId;
use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::atomic::{self, AtomicUsize, Ordering};

use super::cell::ErasedCell;
use super::error::Error;
use super::segment::{ComputeFn, Segment};

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

#[repr(transparent)]
struct SyncCell<T> {
    cell: UnsafeCell<T>,
}

impl<T> std::ops::Deref for SyncCell<T> {
    type Target = UnsafeCell<T>;

    fn deref(&self) -> &Self::Target {
        &self.cell
    }
}

impl<T> From<T> for SyncCell<T> {
    fn from(v: T) -> Self {
        Self {
            cell: UnsafeCell::new(v),
        }
    }
}

// # Safety
//
// * Every value a slot pointer targets is `Sync` (enforced by the typed
//   layer's `RefPort`/`RefPorts` leaf bounds), so concurrent consumers may share
//   `&T` derefs; states themselves are only ever accessed exclusively (one
//   worker per node per generation) and need just `Send`;
// * Each slot has exactly one writer node (its producer), so the `compute`s
//   that run concurrently within one `stabilize` write disjoint slots; and
// * A node runs only once its `counter` reaches 0 -- after every dirty
//   predecessor's `compute` has finished and the atomic decrement has
//   published those writes -- so reads never race a concurrent write.
unsafe impl<T> Sync for SyncCell<T> {}

struct Node {
    compute_fn: ComputeFn,
    input_range: Range<usize>,
    output_range: Range<usize>,
    state: ErasedCell,
}

/// Builder for the core layer [`Graph`].
pub struct Builder {
    input_flags: Vec<SyncCell<bool>>,
    input_ptrs: Vec<SyncCell<*const ()>>,
    output_flags: Vec<SyncCell<bool>>,
    output_ptrs: Vec<SyncCell<*const ()>>,
    output_type_ids: Vec<TypeId>,
    input_from_outputs: Adjacency,
    context: ErasedCell,
    nodes: Vec<Node>,
}

impl Builder {
    pub fn new(context: ErasedCell) -> Self {
        Self {
            input_flags: Vec::new(),
            input_ptrs: Vec::new(),
            output_flags: Vec::new(),
            output_ptrs: Vec::new(),
            output_type_ids: Vec::new(),
            input_from_outputs: Adjacency::new(),
            context,
            nodes: Vec::new(),
        }
    }

    pub fn slot_type_id(&self, index: usize) -> TypeId {
        self.output_type_ids[index]
    }

    pub fn slot_flag(&self, index: usize) -> bool {
        unsafe { *self.output_flags[index].get() }
    }

    pub fn slot_ptr(&self, index: usize) -> *const () {
        unsafe { *self.output_ptrs[index].get() }
    }

    pub fn context(&self) -> &ErasedCell {
        &self.context
    }

    pub fn push(
        &mut self,
        segment: Segment,
        input_indices: &[usize],
    ) -> Result<(usize, Range<usize>), Error> {
        let (input_types, output_types, compute_fn, state, output_flags, output_ptrs) =
            segment.into_parts();

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
        if output_arity != output_ptrs.len() || output_arity != output_flags.len() {
            return Err(Error::OutputArity {
                expected: output_arity,
                actual: output_ptrs.len(),
            });
        }
        self.output_type_ids.extend(output_types);

        let input_begin = self.input_ptrs.len();
        let output_begin = self.output_ptrs.len();
        for &s in input_indices {
            self.input_flags.push(self.slot_flag(s).into());
            self.input_ptrs.push(self.slot_ptr(s).into());
            self.input_from_outputs.push(&[s]);
        }
        self.output_flags
            .extend(output_flags.iter().map(|&f| f.into()));
        self.output_ptrs
            .extend(output_ptrs.iter().map(|&p| p.into()));
        let input_range = input_begin..self.input_ptrs.len();
        let output_range = output_begin..self.output_ptrs.len();

        let node_index = self.nodes.len();
        self.nodes.push(Node {
            compute_fn,
            input_range,
            output_range: output_range.clone(),
            state,
        });
        Ok((node_index, output_range))
    }

    pub fn build(self) -> Graph {
        let Builder {
            input_flags,
            input_ptrs,
            output_flags,
            output_ptrs,
            output_type_ids,
            input_from_outputs,
            context,
            nodes,
        } = self;

        let input_flags = input_flags.into_boxed_slice();
        let input_ptrs = input_ptrs.into_boxed_slice();
        let output_flags = output_flags.into_boxed_slice();
        let output_ptrs = output_ptrs.into_boxed_slice();
        let output_type_ids = output_type_ids.into_boxed_slice();
        let nodes = nodes.into_boxed_slice();

        for f in &input_flags {
            unsafe { *f.get() = false };
        }
        for f in &output_flags {
            unsafe { *f.get() = false };
        }

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
            input_flags,
            input_ptrs,
            output_flags,
            output_ptrs,
            output_type_ids,
            output_to_inputs,
            node_to_nodes,
            context,
            nodes,
            counters,
            roots: Vec::new(),
            dirty: Vec::new(),
            is_dirty,
            poisoned: false,
        }
    }
}

/// The type-erased core of a graph.
pub struct Graph {
    input_flags: Box<[SyncCell<bool>]>,
    input_ptrs: Box<[SyncCell<*const ()>]>,
    output_flags: Box<[SyncCell<bool>]>,
    output_ptrs: Box<[SyncCell<*const ()>]>,
    output_type_ids: Box<[TypeId]>,
    output_to_inputs: Adjacency,
    node_to_nodes: Adjacency,
    context: ErasedCell,
    nodes: Box<[Node]>,
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

    pub fn slot_flag(&self, index: usize) -> bool {
        assert!(!self.poisoned, "cannot access poisoned graph.");
        assert!(self.dirty.is_empty(), "cannot read unstabilized graph.");
        unsafe { *self.output_flags[index].get() }
    }

    pub fn slot_ptr(&self, index: usize) -> *const () {
        assert!(!self.poisoned, "cannot access poisoned graph.");
        assert!(self.dirty.is_empty(), "cannot read unstabilized graph.");
        unsafe { *self.output_ptrs[index].get() }
    }

    pub fn context(&self) -> &ErasedCell {
        &self.context
    }

    pub fn context_mut(&mut self) -> &mut ErasedCell {
        &mut self.context
    }

    pub fn state_mut(&mut self, index: usize) -> &mut ErasedCell {
        assert!(!self.poisoned, "cannot access poisoned graph.");
        if !self.is_dirty[index] {
            self.is_dirty[index] = true;
            self.dirty.push(index);
        }
        &mut self.nodes[index].state
    }

    pub fn stabilize(&mut self, pool: &mut crate::pool::Pool) {
        assert!(!self.poisoned, "cannot access poisoned graph.");
        self.poisoned = true;

        // Discover the dirty cone using breadth-first search, accumulating in
        // `counter` each cone node's number of dirty predecessors.
        let mut cursor = 0;
        while cursor < self.dirty.len() {
            let i = self.dirty[cursor];
            for &j in self.node_to_nodes.get(i) {
                if !self.is_dirty[j] {
                    self.is_dirty[j] = true;
                    self.dirty.push(j);
                }
                *self.counters[j].get_mut() += 1;
            }
            cursor += 1;
        }

        // A cone node with no dirty predecessor (`counter == 0` -- only ever a
        // poked source) is a *root*: the only kind of node that may run
        // immediately. Scanning `dirty` keeps this proportional to the cone.
        for &i in &self.dirty {
            if *self.counters[i].get_mut() == 0 {
                self.roots.push(i);
            }
        }

        // Data flow over the cone on the work-stealing pool: seed the roots,
        // each finished node releases the successors whose counter hits 0.
        // `run` blocks until the batch drains (counters return to 0, ready for
        // the next generation). Tasks are plain `usize`, so this allocates
        // nothing per node.
        pool.run(self.roots.iter().copied(), |i, scope| {
            // Run node `i` compute function.
            let compute_fn = self.nodes[i].compute_fn;
            let in_range = self.nodes[i].input_range.clone();
            let out_range = self.nodes[i].output_range.clone();
            let in_flags = cell_slice(&self.input_flags[in_range.clone()]);
            let in_ptrs = cell_slice(&self.input_ptrs[in_range]);
            let out_flags = cell_slice_mut(&self.output_flags[out_range.clone()]);
            let out_ptrs = cell_slice_mut(&self.output_ptrs[out_range]);
            let context = &self.context;
            let state = &self.nodes[i].state;
            unsafe {
                compute_fn(
                    in_flags,
                    in_ptrs,
                    out_flags,
                    out_ptrs,
                    context.get(),
                    state.get(),
                )
            };

            // Scatter this node's fresh output slots into every consumer's
            // input slot unconditionally. Load-bearing for pointer/reference
            // safety on data inside slots.
            for s in self.nodes[i].output_range.clone() {
                for &t in self.output_to_inputs.get(s) {
                    unsafe {
                        *self.input_flags[t].get() = *self.output_flags[s].get();
                        *self.input_ptrs[t].get() = *self.output_ptrs[s].get();
                    }
                }
            }

            // Report each successor whose last dirty predecessor was `i` --
            // i.e. whose counter reached 0. The writes above happens-before
            // the atomic decrement, so a successor reported ready has observed
            // every dirty predecessor's output.
            for &j in self.node_to_nodes.get(i) {
                // Release publishes this node's writes into the modification
                // order; the Acquire fence on the 0-transition then makes
                // *every* dirty predecessor's writes visible before `j` is
                // released. This is the `Arc`-drop ordering.
                if self.counters[j].fetch_sub(1, Ordering::Release) == 1 {
                    atomic::fence(Ordering::Acquire);
                    scope.spawn(j);
                }
            }
        });
        self.roots.clear();

        // Reset the per-generation notify flags and cone membership for the next
        // round. (Counters already returned to 0 as the cone drained.)
        for i in self.dirty.iter().copied() {
            self.is_dirty[i] = false;
            for s in self.nodes[i].input_range.clone() {
                unsafe { *self.input_flags[s].get() = false };
            }
            for s in self.nodes[i].output_range.clone() {
                unsafe { *self.output_flags[s].get() = false };
            }
        }
        self.dirty.clear();

        // Clear poison flag on clean exit.
        self.poisoned = false;
    }
}

// View a slice of `SyncCell<T>` as a raw slice of `T`, immutably or mutably.
// Layout guarantees of `SyncCell` and `UnsafeCell` make this safe.
// Later casting of disjoint raw slices to mutable slices does not violate
// borrowing models; tested under Miri.
fn cell_slice<T>(cells: &[SyncCell<T>]) -> *const [T] {
    std::ptr::slice_from_raw_parts(cells.as_ptr() as *const T, cells.len())
}

fn cell_slice_mut<T>(cells: &[SyncCell<T>]) -> *mut [T] {
    std::ptr::slice_from_raw_parts_mut(cells.as_ptr() as *mut T, cells.len())
}
