//! Core type-erased graph: nodes and DAG dispatch.
//!
//! [`Graph`] owns [`Node`]s and implements topological flush via a min-heap.
//! Each flush cycle maintains an `incoming` vector of per-node lists that
//! records *which input positions* produced — exposed to operators via
//! [`Notify::input_produced`].

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::time::Instant;
use crate::types::Notify;

use super::node::Node;

/// Untyped DAG owning [`Node`]s and implementing topological flush.
///
/// # Invariants
///
/// * `incoming.len() == nodes.len()`.
/// * `incoming[i].is_empty()` if and only if `i` is **not** in `heap`.
/// * Node indices encode topological order: if node `j` has node `i` as an
///   input (via its operator state's `input_ptrs`), then `i < j`.
/// * Edges: `nodes[i].trigger_edges` contains `(downstream, input_pos)`
///   pairs which should be notified when node `i` updates.
pub(super) struct Graph {
    /// Type-erased nodes.
    pub nodes: Vec<Node>,
    /// Per-node incoming input positions that produced in the current
    /// flush cycle.  Non-empty iff the node is pending in the heap.
    incoming: Vec<Vec<usize>>,
    /// Min-heap of node indices.
    heap: BinaryHeap<Reverse<usize>>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            incoming: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Append a node.  Returns its index (= topological rank).
    pub fn add_node(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.incoming.push(Vec::new());
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

    /// Propagate updates through the DAG.
    ///
    /// For each updated source node, schedules its downstream operator nodes
    /// onto a min-heap keyed by node index (= topological order).  Each
    /// operator is invoked with a [`Notify`] context; if it produces output,
    /// its downstream nodes are scheduled in turn.
    pub fn flush(&mut self, timestamp: Instant, updated_sources: &[usize]) {
        let Self {
            nodes,
            incoming,
            heap,
        } = self;

        // Seed the min-heap from updated source nodes' edges.
        for &i in updated_sources {
            for &(j, input_pos) in &nodes[i].trigger_edges {
                let was_empty = incoming[j].is_empty();
                incoming[j].push(input_pos);
                if was_empty {
                    heap.push(Reverse(j));
                }
            }
        }

        // Process in topological order (node index IS topological rank).
        while let Some(Reverse(i)) = heap.pop() {
            let node = &nodes[i];
            let did_produce = if let Some(state) = node.operator_state() {
                let num_inputs = state.input_node_indices().len();
                let notify = Notify::new(&incoming[i], num_inputs);
                // SAFETY: all pointers validated at node construction time.
                unsafe { state.compute(node.value_ptr, timestamp, &notify) }
            } else {
                false
            };

            // Clear incoming after compute (safe: topological order guarantees
            // no upstream node can push here after this point).
            incoming[i].clear();

            if did_produce {
                for &(j, input_pos) in &nodes[i].trigger_edges {
                    let was_empty = incoming[j].is_empty();
                    incoming[j].push(input_pos);
                    if was_empty {
                        heap.push(Reverse(j));
                    }
                }
            }
        }
    }
}
