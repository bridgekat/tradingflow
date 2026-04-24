//! Type-erased wavefront graph: nodes and compute dispatch.

use std::any::TypeId;

use crate::operator::ComputeFn;

use super::storage::VersionedRing;

/// A node in the wavefront graph.
///
/// For a stateful operator, both `current_state_ptr` and
/// `current_output_ptr` carry across ticks: tick `t+1` clones from
/// the values produced by tick `t`.  For a stateless operator, each
/// tick clones fresh from `init_state_ptr` / `init_output_ptr`.
pub(super) struct WavefrontNode {
    /// Global node index.
    pub index: usize,

    /// Versioned output history (one entry per completed tick).
    pub versioned: VersionedRing,

    // -- Init values (never change after construction) --
    pub init_state_ptr: *mut u8,
    pub init_output_ptr: *mut u8,

    // -- Current values (carried forward for stateful nodes) --
    pub current_state_ptr: *mut u8,
    pub current_output_ptr: *mut u8,

    // -- Clone / drop functions --
    pub state_clone_fn: unsafe fn(*const u8) -> *mut u8,
    pub state_drop_fn: unsafe fn(*mut u8),
    pub output_clone_fn: unsafe fn(*const u8) -> *mut u8,
    pub output_drop_fn: unsafe fn(*mut u8),

    /// TypeId of the output (used for runtime type checking).
    #[allow(dead_code)]
    pub type_id: TypeId,

    /// Whether this is a source node.
    pub is_source: bool,

    /// Monomorphised compute function pointer.
    pub compute_fn: ComputeFn,

    /// Indices of upstream input nodes.
    pub input_indices: Box<[usize]>,

    /// Downstream edges: `(downstream_idx, input_position)`.
    pub trigger_edges: Vec<(usize, usize)>,

    /// Whether the operator is stateful.
    pub is_stateful: bool,
}

impl Drop for WavefrontNode {
    fn drop(&mut self) {
        // Drop current first (must be distinct from init).
        if self.current_output_ptr != self.init_output_ptr {
            unsafe { (self.output_drop_fn)(self.current_output_ptr) };
        }
        if !self.init_output_ptr.is_null() {
            unsafe { (self.output_drop_fn)(self.init_output_ptr) };
        }
        if self.current_state_ptr != self.init_state_ptr {
            unsafe { (self.state_drop_fn)(self.current_state_ptr) };
        }
        if !self.init_state_ptr.is_null() {
            unsafe { (self.state_drop_fn)(self.init_state_ptr) };
        }
    }
}

// SAFETY: Node owns its heap allocations, all types satisfy Send.
unsafe impl Send for WavefrontNode {}

/// The wavefront computation graph.
pub(super) struct WavefrontGraph {
    pub nodes: Vec<WavefrontNode>,
    /// Source node indices.
    pub source_indices: Vec<usize>,
}

impl WavefrontGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            source_indices: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: WavefrontNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    pub fn add_trigger_edge(&mut self, from: usize, to: usize, input_pos: usize) {
        self.nodes[from].trigger_edges.push((to, input_pos));
    }
}
