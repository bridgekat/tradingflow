//! Wavefront graph — owns [`Node`]s and the trigger-edge index.
//!
//! Registration preserves the "node index = topological rank" invariant:
//! every call to [`add_operator`](Graph::add_operator) appends a node
//! whose declared upstreams all have strictly smaller indices.  The
//! scheduler relies on this to ensure correctness of the per-tick
//! readiness counters (upstreams finish signalling before downstreams can
//! reach the readiness edge).

use std::any::TypeId;
use std::sync::Arc;

use super::super::data::Instant;
use super::super::operator::ErasedOperator;
use super::super::source::ErasedSource;
use super::node::{Node, TriggerEdge};

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) source_indices: Vec<usize>,
    pipeline_width: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            source_indices: Vec::new(),
            pipeline_width: 8,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn pipeline_width(&self) -> usize {
        self.pipeline_width
    }

    /// Register a type-erased [`Source`](crate::source::Source).
    pub fn add_source<T: Clone + Send + Sync + 'static>(
        &mut self,
        erased: ErasedSource,
        queue_cap: usize,
        pipeline_width: usize,
    ) -> usize {
        assert_eq!(
            erased.output_type_id(),
            TypeId::of::<T>(),
            "add_source: output type id mismatch",
        );
        self.pipeline_width = pipeline_width;
        let poll_fn = erased.poll_fn();
        let write_fn = erased.write_fn();
        let rx_drop_fn = erased.rx_drop_fn();
        let output_drop_fn = erased.output_drop_fn();
        let (hist_rx_ptr, live_rx_ptr, output_scratch_ptr) = erased.init(Instant::MIN);
        let template: T = unsafe { (&*(output_scratch_ptr as *const T)).clone() };
        let idx = self.nodes.len();
        let node = Node::new_source::<T>(
            idx,
            hist_rx_ptr,
            live_rx_ptr,
            poll_fn,
            write_fn,
            rx_drop_fn,
            output_scratch_ptr,
            template,
            output_drop_fn,
            queue_cap,
        );
        self.nodes.push(node);
        self.source_indices.push(idx);
        idx
    }

    /// Register a type-erased [`Operator`].
    pub fn add_operator<T: Clone + Send + Sync + 'static>(
        &mut self,
        erased: ErasedOperator,
        input_indices: &[usize],
        queue_cap: usize,
        pipeline_width: usize,
    ) -> usize {
        assert_eq!(
            erased.output_type_id(),
            TypeId::of::<T>(),
            "add_operator: output type id mismatch",
        );
        self.pipeline_width = pipeline_width;
        // Validate topological order.
        for &idx in input_indices {
            assert!(
                idx < self.nodes.len(),
                "invalid input index {idx}: out of range",
            );
        }
        // Validate input type ids against upstream outputs.
        for (pos, (&idx, &declared_tid)) in input_indices
            .iter()
            .zip(erased.input_type_ids().iter())
            .enumerate()
        {
            let actual_tid = self.nodes[idx].output_type_id;
            assert_eq!(
                declared_tid, actual_tid,
                "type mismatch at input position {pos}: operator declares one type; upstream node {idx} outputs another",
            );
        }

        // Collect input pointers for init.  We use each upstream's
        // latest-at-or-before(MIN) slot — which is the seed_initial value
        // (the upstream's own template).  This matches the legacy runtime
        // semantic where operators init from the upstream's initial
        // output.
        let mut input_ptrs: Vec<*const u8> = Vec::with_capacity(input_indices.len());
        let mut borrow_guards: Vec<super::node::BorrowGuard> =
            Vec::with_capacity(input_indices.len());
        for &idx in input_indices {
            let (p, g) = self.nodes[idx]
                .output
                .borrow(Instant::MIN)
                .expect("upstream has no seed value");
            input_ptrs.push(p.0);
            borrow_guards.push(g);
        }

        let output_type_id = erased.output_type_id();
        let compute_fn = erased.compute_fn();
        let clone_fn = erased.clone_fn();
        let state_drop_fn = erased.state_drop_fn();
        let input_type_ids: Box<[TypeId]> = erased.input_type_ids().into();
        let arity = input_indices.len();

        // Init the operator: gives (state_ptr, output_template_ptr).
        let (state_ptr, output_tpl_ptr) = unsafe { erased.init(&input_ptrs, Instant::MIN) };

        // Drop the input borrow guards now that init is done.
        drop(borrow_guards);

        debug_assert_eq!(
            output_type_id,
            TypeId::of::<T>(),
            "caller-declared T must match erased operator output type"
        );

        // Unbox the template to move ownership into the store.
        let template: T = unsafe { *Box::from_raw(output_tpl_ptr as *mut T) };
        let input_edges: Box<[(usize, usize)]> = input_indices
            .iter()
            .enumerate()
            .map(|(pos, &upstream)| (upstream, pos))
            .collect();

        // Count how many of this node's direct upstreams participate in
        // the per-tick wavefront (non-Const upstreams).  Const operators
        // have direct_upstream_count == 0 and their
        // `participates_in_wavefront == false`.
        let effective_upstream_count = input_indices
            .iter()
            .filter(|&&up| self.nodes[up].participates_in_wavefront)
            .count();

        let idx = self.nodes.len();
        let mut node = Node::new_operator::<T>(
            idx,
            compute_fn,
            clone_fn,
            state_ptr,
            state_drop_fn,
            template,
            input_edges,
            input_type_ids,
            arity,
            queue_cap,
            pipeline_width,
        );
        node.effective_upstream_count = effective_upstream_count;
        self.nodes.push(node);

        // Install trigger-edges only from wavefront-participating
        // upstreams.  Const upstreams never fire, so installing their
        // edges would be dead code.
        for (pos, &upstream) in input_indices.iter().enumerate() {
            if self.nodes[upstream].participates_in_wavefront {
                self.nodes[upstream].trigger_edges.push(TriggerEdge {
                    downstream: idx,
                    input_pos: pos,
                });
            }
        }
        idx
    }

    /// Access a node's latest committed value as a fresh `Arc<T>`.
    ///
    /// Safe only when no concurrent writer is active (e.g. after
    /// [`Scenario::run`](super::Scenario::run) returns).
    pub fn value<T: Clone + Send + Sync + 'static>(&self, idx: usize) -> Option<Arc<T>> {
        let node = &self.nodes[idx];
        assert_eq!(
            node.output_type_id,
            TypeId::of::<T>(),
            "value(): type mismatch at node {idx}",
        );
        let (_ptr, guard) = node.output.borrow(Instant::MAX)?;
        let slot_arc: Arc<super::super::queue::Slot<T>> = extract_slot_arc::<T>(guard)?;
        Some(Arc::new(slot_arc.value.clone()))
    }

    /// Finalise: freeze trigger_edges into boxed slices for the scheduler.
    pub fn seal(&mut self) {
        for node in &mut self.nodes {
            node.trigger_edges.shrink_to_fit();
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Downcast a [`BorrowGuard`] holding an `Arc<Slot<T>>` back to the
/// concrete `Arc<Slot<T>>`.
fn extract_slot_arc<T: Send + Sync + 'static>(
    guard: super::node::BorrowGuard,
) -> Option<Arc<super::super::queue::Slot<T>>> {
    let any = guard.into_any()?;
    Arc::downcast::<super::super::queue::Slot<T>>(any).ok()
}
