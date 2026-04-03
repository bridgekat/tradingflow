//! Core type-erased graph: nodes and DAG dispatch.
//!
//! [`Graph`] owns [`Node`]s and implements topological flush via a min-heap.
//! Each flush cycle maintains a `produced` flags vector (parallel to `nodes`)
//! that tracks which nodes produced new output.  These flags are exposed to
//! operators through the [`Notify`] context, enabling zero-cost per-input
//! update checks.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::operator::Notify;

use super::node::Node;

/// Untyped DAG owning [`Node`]s and implementing topological flush.
///
/// # Invariants
///
/// * `pending.len() == nodes.len() == produced.len()`.
/// * `pending[i] == true` if and only if `i` is currently in `heap`.
/// * `produced[i]` is `true` when node `i` has produced output in the
///   current flush cycle.
/// * Node indices encode topological order: if node `j` has node `i` as an
///   input (via its operator state's `input_ptrs`), then `i < j`.
/// * Edges: `nodes[i].trigger_edges` contains indices of nodes which should
///   be notified by updates from node `i`.
pub(super) struct Graph {
    /// Type-erased nodes.
    pub nodes: Vec<Node>,
    /// Pending update flags, parallel to `nodes`.
    pending: Vec<bool>,
    /// Produced flags for the current flush cycle, parallel to `nodes`.
    produced: Vec<bool>,
    /// Indices of nodes whose `produced` flag was set this cycle
    /// (used to clear only the dirty flags at the end of flush).
    produced_list: Vec<usize>,
    /// Min-heap of node indices.
    heap: BinaryHeap<Reverse<usize>>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            pending: Vec::new(),
            produced: Vec::new(),
            produced_list: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Append a node.  Returns its index (= topological rank).
    pub fn add_node(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.pending.push(false);
        self.produced.push(false);
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
    /// operator is invoked with a [`Notify`] context; if it produces output,
    /// its downstream nodes are scheduled in turn.
    pub fn flush(&mut self, timestamp: i64, updated_sources: &[usize]) {
        // Destructure for split borrows: `produced` is shared via `Notify`
        // while `pending`/`heap` are mutated for scheduling.
        let Self {
            nodes,
            pending,
            produced,
            produced_list,
            heap,
        } = self;

        // Seed the min-heap from updated source nodes' edges.
        for &i in updated_sources {
            produced[i] = true;
            produced_list.push(i);
            for &j in &nodes[i].trigger_edges {
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
                let notify = Notify::new(produced, state.input_node_indices());
                // SAFETY: all pointers validated at node construction time.
                unsafe { state.compute(node.value_ptr, timestamp, &notify) }
            } else {
                false
            };

            if did_produce {
                produced[i] = true;
                produced_list.push(i);
                for &j in &nodes[i].trigger_edges {
                    if !pending[j] {
                        pending[j] = true;
                        heap.push(Reverse(j));
                    }
                }
            }
        }

        // Clear produced flags set.
        for &i in produced_list.iter() {
            produced[i] = false;
        }
        produced_list.clear();
    }
}
