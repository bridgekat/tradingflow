//! Type-erased node and operator closure for the DAG graph.
//!
//! # Invariants
//!
//! * `type_id == TypeId::of::<T>()` where `T` is the node's value type.
//! * `value` is a valid, non-null pointer to a heap-allocated `T`.
//! * If `closure` is `Some`: `state` is a valid pointer to `Op::State`;
//!   each `input_ptrs[i]` points to a valid value that outlives this node;
//!   `compute_fn` is monomorphised for the correct operator type.
//! * `trigger_edges[i]` are valid node indices in the owning `Graph`.
//!
//! # Safety boundary
//!
//! `Node` construction and access do not guarantee the invariants above.
//! These must be maintained throughout the [`scenario`][super] module.

use std::any::TypeId;

use crate::operator::Operator;
use crate::types::InputKinds;

// ---------------------------------------------------------------------------
// Function pointer types
// ---------------------------------------------------------------------------

/// Type-erased compute function.
/// Returns `true` if an output value was produced, `false` to skip propagation.
///
/// Arguments:
///
/// * `input_ptrs` — `&[*const u8]` pointing to input values.
/// * `output` — `*mut u8` pointing to the output value.
/// * `state` — `*mut u8` pointing to the operator's `State`.
/// * `timestamp` — flush timestamp.
pub(super) type ComputeFn = unsafe fn(&[*const u8], *mut u8, *mut u8, i64) -> bool;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

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

// SAFETY: `Node` owns the heap allocation behind `value`.
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
            value_drop_fn: drop_fn::<T>,
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // Drop the closure's state first (if any).
        if let Some(ref closure) = self.closure {
            // SAFETY: `state` was allocated by `Box::into_raw`.
            unsafe { (closure.state_drop_fn)(closure.state) };
        }
        // Drop the value.
        // SAFETY: `value` was allocated by `Box::into_raw` in `new`.
        unsafe { (self.value_drop_fn)(self.value) };
    }
}

// ---------------------------------------------------------------------------
// Closure
// ---------------------------------------------------------------------------

/// Type-erased operator closure attached to a [`Node`].
///
/// See [module-level docs](self) for layout and invariants.
pub(super) struct Closure {
    /// Monomorphised compute function.
    compute_fn: ComputeFn,
    /// Pre-collected pointers to input values.
    input_ptrs: Box<[*const u8]>,
    /// Heap-allocated operator state (`Box::into_raw(Box::new(state))`).
    pub(super) state: *mut u8,
    /// Drop the state: `drop(Box::from_raw(ptr as *mut State))`.
    pub(super) state_drop_fn: unsafe fn(*mut u8),
}

impl Closure {
    /// Create a [`Closure`] from raw components.
    ///
    /// Used by the bridge to attach Python operator callbacks.
    #[cfg(feature = "python")]
    pub fn new(
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

    /// Build a [`Closure`] for an operator whose state has already been
    /// created (via [`Operator::init`]).
    pub fn from_state<O: Operator>(state: O::State, input_ptrs: Box<[*const u8]>) -> Closure {
        Closure {
            compute_fn: compute_fn::<O>,
            input_ptrs,
            state: Box::into_raw(Box::new(state)) as *mut u8,
            state_drop_fn: drop_fn::<O::State>,
        }
    }

    /// Invokes the closure's compute function with the given output pointer
    /// and timestamp.
    ///
    /// # Safety
    ///
    /// * Each `input_ptrs[i]` must point to a valid value of the type
    ///   expected by `O::Inputs` at position `i`.
    /// * `output_ptr` must point to a valid `O::Output`.
    /// * `output_ptr` must not alias any `input_ptrs[i]`.
    pub unsafe fn compute(&self, output_ptr: *mut u8, timestamp: i64) -> bool {
        unsafe { (self.compute_fn)(&self.input_ptrs, output_ptr, self.state, timestamp) }
    }
}

// SAFETY: `Closure` owns the heap allocation behind `state`.
unsafe impl Send for Closure {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Drop a heap-allocated `T`.
///
/// # Safety
///
/// `ptr` must have been created by `Box::into_raw(Box::new(..))` for type `T`.
unsafe fn drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}

/// Type-erased compute entry point, monomorphised per operator type.
///
/// # Safety
///
/// * Each `input_ptrs[i]` must point to a valid value of the type expected
///   by `O::Inputs` at position `i`.
/// * `output_ptr` must point to a valid `O::Output`.
/// * `state_ptr` must point to a valid `O::State`.
/// * `output_ptr` must not alias any `input_ptrs[i]`.
unsafe fn compute_fn<O: Operator>(
    input_ptrs: &[*const u8],
    output_ptr: *mut u8,
    state_ptr: *mut u8,
    timestamp: i64,
) -> bool {
    unsafe {
        let inputs = <O::Inputs as InputKinds>::from_ptrs(input_ptrs);
        let output = &mut *(output_ptr as *mut O::Output);
        let state = &mut *(state_ptr as *mut O::State);
        O::compute(state, inputs, output, timestamp)
    }
}
