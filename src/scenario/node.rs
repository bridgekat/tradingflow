//! Type-erased node and closure for the DAG graph.
//!
//! # Key types
//!
//! * [`Node`] — a type-erased DAG node owning a value and optionally a
//!   [`Closure`].
//! * [`Closure`] — an installed operator: input pointers and state bound to
//!   a compute function.
//!
//! # Invariants
//!
//! * `Node::type_id == TypeId::of::<T>()` where `T` is the node's value type.
//! * `Node::value` is a valid, non-null pointer to a heap-allocated `T`.
//! * If `closure` is `Some`: `state` is a valid pointer; each `input_ptrs[i]`
//!   points to a valid value that outlives this node; `compute_fn` is
//!   compatible with the types.
//! * `trigger_edges[i]` are valid node indices in the owning `Graph`.

use std::any::TypeId;

use crate::operator::{ComputeFn, ErasedOperator};

// ===========================================================================
// Node
// ===========================================================================

/// Type-erased DAG node: owns a value and optionally a [`Closure`].
///
/// See [module-level docs](self) for layout and invariants.
pub(super) struct Node {
    /// `TypeId::of::<T>()` for the value type `T`.
    pub type_id: TypeId,
    /// Heap-allocated value `T` (via `Box::into_raw`).
    pub value: *mut u8,
    /// Operator closure, or `None` for source / bare nodes.
    pub closure: Option<Closure>,
    /// Downstream node indices that are triggered when this node updates.
    pub trigger_edges: Vec<usize>,
    /// Drop the value: `drop(Box::from_raw(ptr as *mut T))`.
    value_drop_fn: unsafe fn(*mut u8),
}

// SAFETY: `Node` owns the heap allocation behind `value`, which points to
// a type satisfying `Send`.
unsafe impl Send for Node {}

impl Node {
    /// Create a new bare node (no closure) for an arbitrary value type.
    ///
    /// The value is heap-allocated; the returned `Node` owns it.
    pub fn new<T: Send + 'static>(value: T) -> Self {
        let value_ptr = Box::into_raw(Box::new(value));
        Self {
            type_id: TypeId::of::<T>(),
            value: value_ptr as *mut u8,
            closure: None,
            trigger_edges: Vec::new(),
            value_drop_fn: erased_drop_fn::<T>,
        }
    }

    /// Create a bare node from a pre-allocated raw value pointer.
    pub fn from_raw_value(
        type_id: TypeId,
        value: *mut u8,
        drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            type_id,
            value,
            closure: None,
            trigger_edges: Vec::new(),
            value_drop_fn: drop_fn,
        }
    }

    /// Create a node from an [`ErasedOperator`].
    ///
    /// Validates input types, calls the deferred init, and attaches the
    /// compute closure.  Panics on arity or `TypeId` mismatch.
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
        let closure = Closure::new(compute_fn, input_ptrs, state_ptr, state_drop_fn);
        Self {
            type_id: output_type_id,
            value: output_ptr,
            closure: Some(closure),
            trigger_edges: Vec::new(),
            value_drop_fn: output_drop_fn,
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        if let Some(ref closure) = self.closure {
            unsafe { (closure.state_drop_fn)(closure.state) };
        }
        unsafe { (self.value_drop_fn)(self.value) };
    }
}

// ===========================================================================
// Closure
// ===========================================================================

/// Type-erased operator closure attached to a [`Node`].
///
/// This is an installed operator: input pointers and state bound to a
/// compute function.
pub(super) struct Closure {
    /// Monomorphised compute function.
    compute_fn: ComputeFn,
    /// Pre-collected pointers to input values.
    input_ptrs: Box<[*const u8]>,
    /// Heap-allocated operator state.
    pub(super) state: *mut u8,
    /// Drop the state.
    pub(super) state_drop_fn: unsafe fn(*mut u8),
}

impl Closure {
    /// Create a closure from raw components.
    fn new(
        compute_fn: ComputeFn,
        input_ptrs: Box<[*const u8]>,
        state: *mut u8,
        state_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            compute_fn,
            input_ptrs,
            state,
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
        unsafe { (self.compute_fn)(self.state, &self.input_ptrs, output_ptr, timestamp) }
    }
}

// SAFETY: `Closure` owns the heap allocation behind `state`, which points to
// a type satisfying `Send`.
unsafe impl Send for Closure {}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
