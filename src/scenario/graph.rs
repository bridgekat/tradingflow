//! Core type-erased graph: nodes and graph dispatch.
//!
//! [`Graph`] owns [`Node`]s and implements topological flush via a min-heap.
//! Each flush cycle maintains a per-node bitset (`incoming_bits`) that
//! records which input positions produced — the compute function reads it
//! via a [`BitRead`](crate::data::BitRead) cursor to build the operator's
//! nested `Produced<'_>` tree.  A parallel `pending` boolean vector tracks
//! whether each node is currently in the heap.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::Instant;

use super::node::Node;

/// Untyped graph owning [`Node`]s and implementing topological flush.
///
/// # Invariants
///
/// * `incoming_bits.len() == pending.len() == nodes.len()`.
/// * `pending[i] == true` iff `i` is currently in `heap` (set on first
///   enqueue, cleared when popped).
/// * All bits in `incoming_bits[i]` are zero after `i`'s compute runs.
/// * Node indices encode topological order: if node `j` has node `i` as an
///   input, then `i < j`.
/// * `nodes[i].trigger_edges` contains `(downstream, input_pos)` pairs to
///   notify when node `i` produces.
pub(super) struct Graph {
    /// Type-erased nodes.
    pub nodes: Vec<Node>,
    /// Per-node bitset of input positions that produced in the current
    /// flush cycle.  Sized `ceil(num_inputs / 64)` `u64` words per node;
    /// empty for source nodes.
    incoming_bits: Vec<Box<[u64]>>,
    /// Whether the corresponding node is currently in `heap`.
    pending: Vec<bool>,
    /// Min-heap of node indices.
    heap: BinaryHeap<Reverse<usize>>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            incoming_bits: Vec::new(),
            pending: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Append a node.  Returns its index (= topological rank).
    pub fn add_node(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        let arity = node
            .operator_state()
            .map(|s| s.input_node_indices().len())
            .unwrap_or(0);
        let words = arity.div_ceil(64);
        self.incoming_bits
            .push(vec![0u64; words].into_boxed_slice());
        self.pending.push(false);
        self.nodes.push(node);
        idx
    }

    /// Add a trigger edge: when `from` is updated, `to` is scheduled at
    /// `input_pos`.
    ///
    /// # Panics
    ///
    /// Panics if `from < to < self.nodes.len()` is not satisfied.
    pub fn add_trigger_edge(&mut self, from: usize, to: usize, input_pos: usize) {
        assert!(from < to, "nodes must be added in topological order");
        assert!(to < self.nodes.len(), "node index out of bounds");
        self.nodes[from].trigger_edges.push((to, input_pos));
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Propagate updates through the graph.
    ///
    /// For each updated source node, schedules its downstream operator nodes
    /// onto a min-heap keyed by node index (= topological order).  Each
    /// operator is invoked with the current bit range of its `incoming_bits`;
    /// if it produces output, its downstream nodes are scheduled in turn.
    pub fn flush(&mut self, timestamp: Instant, updated_sources: &[usize]) {
        let Self {
            nodes,
            incoming_bits,
            pending,
            heap,
        } = self;

        // Seed the min-heap from updated source nodes' edges.
        for &i in updated_sources {
            for &(j, input_pos) in &nodes[i].trigger_edges {
                incoming_bits[j][input_pos / 64] |= 1u64 << (input_pos % 64);
                if !pending[j] {
                    pending[j] = true;
                    heap.push(Reverse(j));
                }
            }
        }

        // Process in topological order (node index IS topological rank).
        while let Some(Reverse(i)) = heap.pop() {
            pending[i] = false;
            let node = &nodes[i];
            let did_produce = if let Some(state) = node.operator_state() {
                let num_inputs = state.input_node_indices().len();
                // SAFETY: all pointers validated at node construction time.
                unsafe {
                    state.compute(node.value_ptr, timestamp, &incoming_bits[i], 0, num_inputs)
                }
            } else {
                false
            };

            // Clear the produced bitset after compute (safe: topological
            // order guarantees no upstream can write here after this point).
            for w in incoming_bits[i].iter_mut() {
                *w = 0;
            }

            if did_produce {
                for &(j, input_pos) in &nodes[i].trigger_edges {
                    incoming_bits[j][input_pos / 64] |= 1u64 << (input_pos % 64);
                    if !pending[j] {
                        pending[j] = true;
                        heap.push(Reverse(j));
                    }
                }
            }
        }
    }
}
