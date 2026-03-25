//! Type-erased node, closure, and erased operator for the DAG graph.
//!
//! # Key types
//!
//! * [`ErasedOperator`] — type-erased, pre-initialized operator ready to be
//!   installed into the DAG via
//!   [`Scenario::add_erased_operator`](super::Scenario::add_erased_operator).
//! * [`Closure`] — an installed operator: input pointers and state bound to
//!   a compute function.
//! * [`Node`] — a type-erased DAG node owning a value and optionally a
//!   [`Closure`].
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

use crate::operator::Operator;
use crate::types::InputTypes;

// ===========================================================================
// Function pointer types
// ===========================================================================

/// Type-erased compute function.
///
/// Returns `true` if an output value was produced, `false` to skip
/// downstream propagation.
///
/// Arguments:
///
/// * `input_ptrs` — `&[*const u8]` pointing to input values.
/// * `output` — `*mut u8` pointing to the output value.
/// * `state` — `*mut u8` pointing to the operator's mutable state.
/// * `timestamp` — flush timestamp.
pub type ComputeFn = unsafe fn(&[*const u8], *mut u8, *mut u8, i64) -> bool;

// ===========================================================================
// ErasedOperator
// ===========================================================================

/// Type-erased, pre-initialized operator ready to be installed into the DAG.
///
/// Holds the operator's state and initial output (already produced by
/// `init`), plus the function pointers needed for compute and cleanup.
///
/// # Lifecycle
///
/// 1. Created via [`from_operator`](ErasedOperator::from_operator) (safe,
///    typed) or [`new`](ErasedOperator::new) (`unsafe`, raw).
/// 2. Consumed by
///    [`Scenario::add_erased_operator`](super::Scenario::add_erased_operator),
///    which transfers the state and output pointers into the DAG.
/// 3. If dropped without being installed, the `Drop` impl cleans up.
pub struct ErasedOperator {
    /// Expected `TypeId` for each input position.
    pub input_type_ids: Box<[TypeId]>,
    /// `TypeId` of the output value.
    pub output_type_id: TypeId,
    /// Heap-allocated operator state (via `Box::into_raw`), or null if
    /// already transferred.
    state: *mut u8,
    /// Heap-allocated initial output value (via `Box::into_raw`), or null
    /// if already transferred.
    output: *mut u8,
    /// Compute function pointer.
    pub(super) compute_fn: ComputeFn,
    /// Drop the state pointer.
    pub(super) state_drop_fn: unsafe fn(*mut u8),
    /// Drop the output pointer.
    pub(super) output_drop_fn: unsafe fn(*mut u8),
}

// SAFETY: all contained pointers are to heap allocations created by the
// current thread; the function pointers are Send by construction.
unsafe impl Send for ErasedOperator {}

impl ErasedOperator {
    /// Construct from raw, pre-initialized components.
    ///
    /// # Safety
    ///
    /// * `state` must be a valid pointer from `Box::into_raw`, or null.
    /// * `output` must be a valid pointer from `Box::into_raw` to a value
    ///   whose `TypeId` matches `output_type_id`, or null.
    /// * `compute_fn` must correctly interpret input pointers as
    ///   `input_type_ids` types, output pointer as `output_type_id` type,
    ///   and state pointer as the type behind `state`.
    /// * `state_drop_fn` must safely drop the `state` pointer.
    /// * `output_drop_fn` must safely drop the `output` pointer.
    pub unsafe fn new(
        input_type_ids: Box<[TypeId]>,
        output_type_id: TypeId,
        state: *mut u8,
        output: *mut u8,
        compute_fn: ComputeFn,
        state_drop_fn: unsafe fn(*mut u8),
        output_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            input_type_ids,
            output_type_id,
            state,
            output,
            compute_fn,
            state_drop_fn,
            output_drop_fn,
        }
    }

    /// Construct from a typed [`Operator`] implementation.
    ///
    /// Calls [`Operator::init`] eagerly with the given input pointers and
    /// timestamp, producing the operator state and initial output value.
    ///
    /// # Safety
    ///
    /// Each `input_ptrs[i]` must point to a valid value whose `TypeId`
    /// matches `O::Inputs::type_ids()` at position `i`.
    pub unsafe fn from_operator<O: Operator>(
        op: O,
        input_ptrs: &[*const u8],
        timestamp: i64,
    ) -> Self {
        let inputs = unsafe { <O::Inputs as InputTypes>::from_ptrs(input_ptrs) };
        let (state, output) = op.init(inputs, timestamp);
        Self {
            input_type_ids: <O::Inputs as InputTypes>::type_ids(input_ptrs.len()),
            output_type_id: TypeId::of::<O::Output>(),
            state: Box::into_raw(Box::new(state)) as *mut u8,
            output: Box::into_raw(Box::new(output)) as *mut u8,
            compute_fn: compute_fn::<O>,
            state_drop_fn: drop_fn::<O::State>,
            output_drop_fn: drop_fn::<O::Output>,
        }
    }

    /// Take ownership of the state and output pointers.
    ///
    /// After this call, `self.state` and `self.output` are null, so the
    /// `Drop` impl becomes a no-op for those fields.
    pub(super) fn take_ptrs(&mut self) -> (*mut u8, *mut u8) {
        let state = std::mem::replace(&mut self.state, std::ptr::null_mut());
        let output = std::mem::replace(&mut self.output, std::ptr::null_mut());
        (state, output)
    }
}

impl Drop for ErasedOperator {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { (self.state_drop_fn)(self.state) };
        }
        if !self.output.is_null() {
            unsafe { (self.output_drop_fn)(self.output) };
        }
    }
}

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

    /// Create a bare node from raw, type-erased components.
    ///
    /// # Safety
    ///
    /// * `value` must be a valid pointer from `Box::into_raw`.
    /// * `type_id` must match the actual type behind `value`.
    /// * `value_drop_fn` must correctly drop the value.
    pub(super) fn from_raw(
        type_id: TypeId,
        value: *mut u8,
        value_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            type_id,
            value,
            closure: None,
            trigger_edges: Vec::new(),
            value_drop_fn,
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
    pub(super) fn new(
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
        unsafe { (self.compute_fn)(&self.input_ptrs, output_ptr, self.state, timestamp) }
    }
}

// SAFETY: `Closure` owns the heap allocation behind `state`.
unsafe impl Send for Closure {}

// ===========================================================================
// Helpers
// ===========================================================================

/// Drop a heap-allocated `T`.
///
/// # Safety
///
/// `ptr` must have been created by `Box::into_raw(Box::new(..))` for type `T`.
pub unsafe fn drop_fn<T>(ptr: *mut u8) {
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
        let inputs = <O::Inputs as InputTypes>::from_ptrs(input_ptrs);
        let output = &mut *(output_ptr as *mut O::Output);
        let state = &mut *(state_ptr as *mut O::State);
        O::compute(state, inputs, output, timestamp)
    }
}
