//! Type-erased node and operator closure for the DAG graph.
//!
//! # Invariants
//!
//! * `type_id == TypeId::of::<T>()` where `T` is the store's scalar type.
//! * `store` is a valid, non-null pointer to a heap-allocated `Store<T>`.
//! * `shape` points into `Store<T>.shape`; valid for the node's lifetime
//!   (the Store is heap-allocated and never moved).
//! * If `closure` is `Some`: `state` is a valid pointer to `Op::State`;
//!   each `input_ptrs[i]` points to a valid `Store<U>` (possibly different
//!   scalar types) that outlives this node; `compute_fn` is monomorphised
//!   for the correct operator type.
//! * `edges[i]` are valid node indices in the owning `Graph`.
//!
//! # Safety boundary
//!
//! `Node` construction and access do not guarantee the invariants above.
//! These must be maintained throughout the [`scenario`][super] module.

use std::any::TypeId;

use crate::operator::Operator;
use crate::store::Store;
use crate::types::{InputKinds, Scalar};

// ---------------------------------------------------------------------------
// Function pointer types
// ---------------------------------------------------------------------------

/// Type-erased compute function.
/// Returns `true` if an output value was produced, `false` to rollback.
///
/// Arguments:
///
/// * `input_ptrs` — `&[*const u8]` pointing to input `Store<T>`'s.
/// * `output_store` — `*mut u8` pointing to the output `Store<T>`.
/// * `state` — `*mut u8` pointing to the operator's `State`.
/// * `timestamp` — flush timestamp for the new element.
pub(super) type ComputeFn = unsafe fn(&[*const u8], *mut u8, *mut u8, i64) -> bool;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// Type-erased DAG node: owns a [`Store`] and optionally a [`Closure`].
///
/// See [module-level docs](self) for layout and invariants.
pub(super) struct Node {
    /// `TypeId::of::<T>()` for the scalar type `T`.
    pub type_id: TypeId,
    /// Heap-allocated `Store<T>` (via `Box::into_raw`).
    pub store: *mut u8,
    /// Points into `Store<T>.shape` (a `Box<[usize]>` inside the
    /// heap-allocated Store).  Valid for the node's lifetime.
    shape: *const [usize],
    /// Operator closure, or `None` for source / bare nodes.
    pub closure: Option<Closure>,
    /// Downstream node indices (nodes whose closures read this node).
    pub edges: Vec<usize>,
    /// Drop the store: `drop(Box::from_raw(ptr as *mut Store<T>))`.
    store_drop_fn: unsafe fn(*mut u8),
}

// SAFETY: `Node` owns the heap allocation behind `store`.
unsafe impl Send for Node {}

impl Node {
    /// Create a new bare node (no closure) for a given scalar type.
    ///
    /// The store is heap-allocated; the returned `Node` owns it.
    pub fn from_store<T: Scalar>(store: Store<T>) -> Self {
        let store_ptr = Box::into_raw(Box::new(store));
        // SAFETY: store_ptr is valid; Store<T>.shape is a Box<[usize]> whose
        // heap allocation is stable for the Store's lifetime.
        let shape_ptr = unsafe { (*store_ptr).shape() as *const [usize] };
        Self {
            type_id: TypeId::of::<T>(),
            store: store_ptr as *mut u8,
            shape: shape_ptr,
            closure: None,
            edges: Vec::new(),
            store_drop_fn: drop_fn::<Store<T>>,
        }
    }

    /// Element shape, borrowed from the underlying `Store<T>`.
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        // SAFETY: `shape` points into a heap-allocated `Store<T>.shape`
        // which is valid for the node's lifetime (Store is behind
        // `Box::into_raw` and only freed in `Drop`).
        unsafe { &*self.shape }
    }

    /// Number of scalars per element (product of shape dimensions).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.shape().iter().product::<usize>()
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // Drop the closure's state first (if any).
        if let Some(ref closure) = self.closure {
            // SAFETY: `state` was allocated by `Box::into_raw` in `new_closure`.
            unsafe { (closure.state_drop_fn)(closure.state) };
        }
        // Drop the store (which also frees the shape that `self.shape` points to).
        // SAFETY: `store` was allocated by `Box::into_raw` in `new_node`.
        unsafe { (self.store_drop_fn)(self.store) };
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
    /// Pre-collected pointers to input `Store<T>`'s.
    input_ptrs: Box<[*const u8]>,
    /// Heap-allocated operator state (`Box::into_raw(Box::new(op.init()))`).
    state: *mut u8,
    /// Drop the state: `drop(Box::from_raw(ptr as *mut State))`.
    state_drop_fn: unsafe fn(*mut u8),
}

impl Closure {
    /// Create a [`Closure`] from raw components.
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

    /// Build a [`Closure`] for an operator.
    ///
    /// # Arguments
    ///
    /// * `input_ptrs` — pointers to input stores, collected at registration.
    /// * `op_state` — the operator's initial state (from [`Operator::init`]).
    pub fn from_operator<O: Operator>(operator: O, input_ptrs: Box<[*const u8]>) -> Closure {
        let state = operator.init();
        Closure {
            compute_fn: compute_fn::<O>,
            input_ptrs,
            state: Box::into_raw(Box::new(state)) as *mut u8,
            state_drop_fn: drop_fn::<O::State>,
        }
    }

    /// Invokes the closure's compute function with the given output store and
    /// timestamp.
    ///
    /// # Safety
    ///
    /// * Each `input_ptrs[i]` must point to a valid `Store` of the type
    ///   expected by `O::Inputs` at position `i`.
    /// * `output_ptr` must point to a valid `Store<O::Scalar>`.
    /// * `state_ptr` must point to a valid `O::State`.
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
/// Pushes the store's default values as a new element, then calls
/// [`Operator::compute`] to overwrite it.  On failure (returns `false`),
/// the element is popped (rollback).
///
/// # Safety
///
/// * Each `input_ptrs[i]` must point to a valid `Store` of the type expected
///   by `O::Inputs` at position `i`.
/// * `output_ptr` must point to a valid `Store<O::Scalar>`.
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
        let store = &mut *(output_ptr as *mut Store<O::Scalar>);
        let state = &mut *(state_ptr as *mut O::State);
        store.push_default(timestamp);
        let output = store.current_view_mut();
        let produced = O::compute(state, inputs, output);
        if produced {
            store.commit();
        } else {
            store.rollback();
        }
        produced
    }
}
