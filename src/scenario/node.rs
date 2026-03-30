//! Type-erased node and its processing state for the DAG graph.
//!
//! # Key types
//!
//! * [`Node`] — a type-erased DAG node owning a value and a [`NodeState`].
//! * [`NodeState`] — classifies a node as a [`Source`](NodeState::Source) or
//!   an [`Operator`](NodeState::Operator).
//! * [`SourceState`] — per-source channel state and function pointers.
//! * [`OperatorState`] — per-operator computation state and function pointers.
//!
//! # Invariants
//!
//! * `Node::type_id == TypeId::of::<T>()` where `T` is the node's value type.
//! * `Node::value_ptr` is a valid, non-null pointer to a heap-allocated `T`.
//! * For [`NodeState::Operator`]: [`OperatorState::state_ptr`] is a valid
//!   pointer; each `input_ptrs[i]` points to a valid value that outlives this
//!   node; `compute_fn` is compatible with the types.
//! * For [`NodeState::Source`]: `hist_rx_ptr` and `live_rx_ptr` are valid
//!   pointers to `PeekableReceiver<(i64, E)>` allocations.
//! * `trigger_edges[i]` are valid node indices in the owning `Graph`.

use std::any::TypeId;

use crate::operator::{ComputeFn, ErasedOperator};
use crate::source::{ErasedSource, PollFn, WriteFn};

// ===========================================================================
// NodeState
// ===========================================================================

/// State of the node based on its role.
pub(super) enum NodeState {
    Source(SourceState),
    Operator(OperatorState),
}

// ===========================================================================
// Node
// ===========================================================================

/// Type-erased DAG node: owns a value and a [`NodeState`].
///
/// See [module-level docs](self) for layout and invariants.
pub(super) struct Node {
    /// `TypeId::of::<T>()` for the value type `T`.
    pub type_id: TypeId,
    /// Heap-allocated value `T` (via `Box::into_raw`).
    pub value_ptr: *mut u8,
    /// Node classification and attached state.
    pub state: NodeState,
    /// Downstream node indices that are triggered when this node updates.
    pub trigger_edges: Vec<usize>,
    /// Drop the value: `drop(Box::from_raw(ptr as *mut T))`.
    value_drop_fn: unsafe fn(*mut u8),
}

// SAFETY: `Node` owns the heap allocation behind `value_ptr`, which points to
// a type satisfying `Send`.
unsafe impl Send for Node {}

impl Node {
    /// Create a source node from an [`ErasedSource`].
    ///
    /// Calls the deferred init and attaches the channel state.
    pub fn from_erased_source(erased: ErasedSource, timestamp: i64) -> Self {
        let output_type_id = erased.output_type_id();
        let poll_fn = erased.poll_fn();
        let write_fn = erased.write_fn();
        let rx_drop_fn = erased.rx_drop_fn();
        let output_drop_fn = erased.output_drop_fn();
        let (hist_rx_ptr, live_rx_ptr, output_ptr) = erased.init(timestamp);
        let state = SourceState::new(hist_rx_ptr, live_rx_ptr, poll_fn, write_fn, rx_drop_fn);
        Self {
            type_id: output_type_id,
            value_ptr: output_ptr,
            state: NodeState::Source(state),
            trigger_edges: Vec::new(),
            value_drop_fn: output_drop_fn,
        }
    }

    /// Create an operator node from an [`ErasedOperator`].
    ///
    /// Validates input types, calls the deferred init, and attaches the
    /// compute state.  Panics on arity or `TypeId` mismatch.
    pub fn from_erased_operator(
        erased: ErasedOperator,
        input_ptrs: Box<[*const u8]>,
        input_type_ids: &[TypeId],
        timestamp: i64,
    ) -> Self {
        assert_eq!(
            input_type_ids.len(),
            erased.input_type_ids().len(),
            "arity mismatch: operator expects {} inputs, got {}",
            erased.input_type_ids().len(),
            input_type_ids.len(),
        );
        for (i, (&expected, &actual)) in erased
            .input_type_ids()
            .iter()
            .zip(input_type_ids)
            .enumerate()
        {
            assert_eq!(expected, actual, "type mismatch at input {i}");
        }
        let output_type_id = erased.output_type_id();
        let compute_fn = erased.compute_fn();
        let state_drop_fn = erased.state_drop_fn();
        let output_drop_fn = erased.output_drop_fn();
        let (state_ptr, output_ptr) = unsafe { erased.init(&input_ptrs, timestamp) };
        let state = OperatorState::new(compute_fn, input_ptrs, state_ptr, state_drop_fn);
        Self {
            type_id: output_type_id,
            value_ptr: output_ptr,
            state: NodeState::Operator(state),
            trigger_edges: Vec::new(),
            value_drop_fn: output_drop_fn,
        }
    }

    /// Returns the [`SourceState`] if this is a source node.
    pub fn source_state(&self) -> Option<&SourceState> {
        match &self.state {
            NodeState::Source(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the [`OperatorState`] if this is an operator node.
    pub fn operator_state(&self) -> Option<&OperatorState> {
        match &self.state {
            NodeState::Operator(s) => Some(s),
            _ => None,
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // `self.state` is dropped automatically, handling Source/Operator state.
        unsafe { (self.value_drop_fn)(self.value_ptr) };
    }
}

// ===========================================================================
// SourceState
// ===========================================================================

/// Indicates which channel (historical or live) an event came from.
#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub enum ChannelKind {
    Hist,
    Live,
}

/// Per-source channel state and function pointers.
///
/// Owns the two channel state allocations (`hist_rx_ptr` and `live_rx_ptr`)
/// and the function pointers needed to interact with them.
pub(super) struct SourceState {
    hist_rx_ptr: *mut u8,
    live_rx_ptr: *mut u8,
    poll_fn: PollFn,
    write_fn: WriteFn,
    rx_drop_fn: unsafe fn(*mut u8),
}

impl SourceState {
    pub(super) fn new(
        hist_rx_ptr: *mut u8,
        live_rx_ptr: *mut u8,
        poll_fn: PollFn,
        write_fn: WriteFn,
        rx_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            hist_rx_ptr,
            live_rx_ptr,
            poll_fn,
            write_fn,
            rx_drop_fn,
        }
    }

    pub fn rx_ptr(&self, kind: ChannelKind) -> *mut u8 {
        match kind {
            ChannelKind::Hist => self.hist_rx_ptr,
            ChannelKind::Live => self.live_rx_ptr,
        }
    }

    pub fn poll_fn(&self) -> PollFn {
        self.poll_fn
    }

    pub fn write_fn(&self) -> WriteFn {
        self.write_fn
    }
}

impl Drop for SourceState {
    fn drop(&mut self) {
        unsafe { (self.rx_drop_fn)(self.hist_rx_ptr) };
        unsafe { (self.rx_drop_fn)(self.live_rx_ptr) };
    }
}

// SAFETY: `SourceState` owns the heap allocations behind `hist_rx_ptr` and
// `live_rx_ptr`, which point to [`PeekableReceiver`] types satisfying `Send`.
unsafe impl Send for SourceState {}

// ===========================================================================
// OperatorState
// ===========================================================================

/// Per-operator computation state and function pointers.
///
/// Owns the heap-allocated [`Operator::State`](crate::Operator::State),
/// pre-collected input pointers, and the monomorphised compute function.
pub(super) struct OperatorState {
    /// Monomorphised compute function.
    compute_fn: ComputeFn,
    /// Pre-collected pointers to input values.
    input_ptrs: Box<[*const u8]>,
    /// Heap-allocated operator state.
    pub(super) state_ptr: *mut u8,
    /// Drop the state.
    state_drop_fn: unsafe fn(*mut u8),
}

impl OperatorState {
    /// Create from raw components.
    fn new(
        compute_fn: ComputeFn,
        input_ptrs: Box<[*const u8]>,
        state_ptr: *mut u8,
        state_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            compute_fn,
            input_ptrs,
            state_ptr,
            state_drop_fn,
        }
    }

    /// Invoke the compute function.
    ///
    /// # Safety
    ///
    /// * Each `input_ptrs[i]` must point to a valid value of the expected type.
    /// * `output_ptr` must point to a valid output value.
    /// * `output_ptr` must not alias any `input_ptrs[i]`.
    pub unsafe fn compute(&self, output_ptr: *mut u8, timestamp: i64) -> bool {
        unsafe { (self.compute_fn)(self.state_ptr, &self.input_ptrs, output_ptr, timestamp) }
    }
}

impl Drop for OperatorState {
    fn drop(&mut self) {
        unsafe { (self.state_drop_fn)(self.state_ptr) };
    }
}

// SAFETY: `OperatorState` owns the heap allocation behind `state_ptr`, which
// points to a type satisfying `Send`.
unsafe impl Send for OperatorState {}
