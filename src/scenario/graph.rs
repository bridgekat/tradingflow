//! Core type-erased graph: nodes and DAG dispatch.
//!
//! [`Graph`] owns [`Node`]s and implements topological flush via a min-heap.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::node::Node;

/// Untyped DAG owning [`Node`]s and implementing topological flush.
///
/// # Invariants
///
/// * `pending.len() == nodes.len()`.
/// * `pending[i] == true` if and only if `i` is currently in `heap`.
/// * Node indices encode topological order: if node `j` has node `i` as an
///   input (via its operator state's `input_ptrs`), then `i < j`.
/// * Edges: `nodes[i].trigger_edges` contains indices of nodes which should
///   be notified by updates from node `i`.
pub(super) struct Graph {
    /// Type-erased nodes.
    pub nodes: Vec<Node>,
    /// Pending update flags, parallel to `nodes`.
    pending: Vec<bool>,
    /// Min-heap of node indices.
    heap: BinaryHeap<Reverse<usize>>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            pending: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Append a node.  Returns its index (= topological rank).
    pub fn add_node(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.pending.push(false);
        idx
    }

    /// Add a trigger edge: when `from` is updated, `to` is scheduled.
    ///
    /// # Panics
    ///
    /// Panics if `from < to < self.nodes.len()` is not satisfied.
    pub fn add_trigger_edge(&mut self, from: usize, to: usize) {
        assert!(from < to, "nodes must be added in topological order");
        assert!(to < self.nodes.len(), "node index out of bounds");
        self.nodes[from].trigger_edges.push(to);
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Propagate updates through the DAG.
    ///
    /// For each updated source node, schedules its downstream operator nodes
    /// onto a min-heap keyed by node index (= topological order).  Each
    /// operator is invoked; if it produces output, its downstream nodes are
    /// scheduled in turn.
    pub fn flush(&mut self, timestamp: i64, updated_sources: &[usize]) {
        // Seed the min-heap from updated source nodes' edges.
        for &i in updated_sources {
            for &j in &self.nodes[i].trigger_edges {
                if !self.pending[j] {
                    self.pending[j] = true;
                    self.heap.push(Reverse(j));
                }
            }
        }

        // Process in topological order (node index IS topological rank).
        while let Some(Reverse(i)) = self.heap.pop() {
            self.pending[i] = false;

            let node = &self.nodes[i];
            let produced = if let Some(state) = node.operator_state() {
                // SAFETY: all pointers validated at node construction time.
                unsafe { state.compute(node.value_ptr, timestamp) }
            } else {
                false
            };

            if produced {
                for &j in &self.nodes[i].trigger_edges {
                    if !self.pending[j] {
                        self.pending[j] = true;
                        self.heap.push(Reverse(j));
                    }
                }
            }
        }
    }
}
