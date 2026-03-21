//! Core type-erased graph: nodes and DAG dispatch.
//!
//! [`Graph`] owns [`Node`]s and implements topological flush via a min-heap.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::node::Node;

/// Untyped DAG graph owning [`Node`]s and implementing topological flush.
///
/// # Memory layout
///
/// ```text
/// Graph {
///     nodes:   Vec<Node>                  // type-erased nodes
///     pending: Vec<bool>                  // pending[i] <=> node i is in the heap
///     heap:    BinaryHeap<Reverse<usize>> // min-heap of node indices
/// }
/// ```
///
/// # Invariants
///
/// * `pending.len() == nodes.len()`.
/// * `pending[i] == true` if and only if `i` is currently in `heap`.
/// * Node indices encode topological order: if node `j` has node `i` as an
///   input (via its closure's `input_ptrs`), then `i < j`.
/// * Edges: `nodes[i].edges` contains indices of nodes whose closures read
///   from node `i`.
///
/// # Safety
///
/// All methods on `Graph` are safe, assuming nodes were correctly constructed
/// (which the typed `Scenario` layer guarantees).
pub(super) struct Graph {
    pub nodes: Vec<Node>,
    pending: Vec<bool>,
    heap: BinaryHeap<Reverse<usize>>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            pending: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Append a node.  Returns its index (= topological rank).
    pub fn push_node(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.pending.push(false);
        idx
    }

    /// Add a directed edge: when `from` is updated, `to` is scheduled.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.nodes[from].edges.push(to);
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Propagate updates through the DAG.
    ///
    /// For each updated source node, schedules its downstream closure nodes
    /// onto a min-heap keyed by node index (= topological order).  Each
    /// closure is invoked; if it produces output, its downstream nodes are
    /// scheduled in turn.
    pub fn flush(&mut self, timestamp: i64, updated_sources: &[usize]) {
        // Seed the min-heap from updated source nodes' edges.
        for &src_idx in updated_sources {
            for &downstream in &self.nodes[src_idx].edges {
                if !self.pending[downstream] {
                    self.pending[downstream] = true;
                    self.heap.push(Reverse(downstream));
                }
            }
        }

        // Process in topological order (node index IS topological rank).
        while let Some(Reverse(node_idx)) = self.heap.pop() {
            self.pending[node_idx] = false;

            let node = &self.nodes[node_idx];
            let produced = if let Some(ref closure) = node.closure {
                // SAFETY: all pointers validated at node construction time.
                unsafe {
                    (closure.compute_fn)(&closure.input_ptrs, node.store, closure.state, timestamp)
                }
            } else {
                false
            };

            if produced {
                for &downstream in &self.nodes[node_idx].edges {
                    if !self.pending[downstream] {
                        self.pending[downstream] = true;
                        self.heap.push(Reverse(downstream));
                    }
                }
            }
        }
    }
}
