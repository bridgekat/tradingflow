//! Session — a single live execution of a [`Scenario`](super::Scenario).
//!
//! A [`Scenario`](super::Scenario) is a *pure definition*: a list of source
//! and operator descriptors plus the wiring between them.  It does not own
//! any node values, channel state, or operator state.  Those live on a
//! [`Session`], built by [`Scenario::build_session`](super::Scenario::build_session)
//! by replaying every registered descriptor's `init` against fresh
//! buffers.
//!
//! All node-value access ([`value`](Session::value),
//! [`value_mut`](Session::value_mut), [`value_ptr`](Session::value_ptr)),
//! manual propagation ([`flush`](Session::flush)) and the async event loop
//! ([`run`](Session::run) / [`run_with_shutdown`](Session::run_with_shutdown))
//! belong to `Session`.  A scenario can drive any number of independent
//! sessions over its lifetime; each starts with freshly allocated state.

use std::any::TypeId;

use crate::Instant;

use super::graph::Graph;
use super::handle::Handle;

/// A live execution of a [`Scenario`](super::Scenario).
///
/// Owns the per-run state — node value buffers, channel receivers,
/// operator state — and exposes the per-run API.  Constructed via
/// [`Scenario::build_session`](super::Scenario::build_session); each
/// session is independent of every other and can be dropped without
/// affecting the underlying definition.
pub struct Session {
    pub(super) graph: Graph,
    pub(super) source_indices: Vec<usize>,
    /// Aggregate estimated event count across all sources, copied from
    /// the [`Scenario`](super::Scenario) at session-build time so the
    /// event loop has it without needing a back-reference.
    pub(super) estimated_event_count: Option<usize>,
}

impl Session {
    /// Immutable access to a node's value.  Panics on TypeId mismatch.
    #[inline(always)]
    pub fn value<T: Send + 'static>(&self, h: Handle<T>) -> &T {
        let node = &self.graph.nodes[h.index()];
        assert_eq!(
            node.type_id,
            TypeId::of::<T>(),
            "type mismatch at node {}",
            h.index(),
        );
        unsafe { &*(node.value_ptr as *const T) }
    }

    /// Mutable access to a node's value.  Panics on TypeId mismatch.
    #[inline(always)]
    pub fn value_mut<T: Send + 'static>(&mut self, h: Handle<T>) -> &mut T {
        let node = &self.graph.nodes[h.index()];
        assert_eq!(
            node.type_id,
            TypeId::of::<T>(),
            "type mismatch at node {}",
            h.index(),
        );
        unsafe { &mut *(node.value_ptr as *mut T) }
    }

    /// Raw value pointer to a node.
    pub fn value_ptr(&self, index: usize) -> *mut u8 {
        self.graph.nodes[index].value_ptr
    }

    /// Aggregate estimated event count across all sources at the time
    /// this session was built.  See
    /// [`Scenario::estimated_event_count`](super::Scenario::estimated_event_count).
    #[inline]
    pub fn estimated_event_count(&self) -> Option<usize> {
        self.estimated_event_count
    }

    /// Propagate updates through the graph for a manually-constructed
    /// flush batch.
    pub fn flush(&mut self, timestamp: Instant, updated_sources: &[usize]) {
        self.graph.flush(timestamp, updated_sources);
    }
}
