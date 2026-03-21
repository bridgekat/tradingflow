//! Type-erased node and operator closure for the DAG graph.
//!
//! # Memory layout
//!
//! ```text
//! Node {
//!     store:      *mut u8             // Box::into_raw(Box::new(Store<T>))
//!     type_id:    TypeId              // TypeId::of::<T>()  (scalar type)
//!     shape:      *const [usize]      // borrows Store<T>.shape (stable heap ptr)
//!     closure:    Option<Closure>     // None for sources, Some for operators
//!     edges:      Vec<usize>          // downstream node indices
//!     drop_store: unsafe fn(*mut u8)  // destructor
//! }
//!
//! Closure {
//!     compute_fn:  ComputeFn                  // monomorphised fn pointer
//!     state:       *mut u8                    // heap-allocated Op::State
//!     input_ptrs:  Box<[*const u8]>           // pointers to input Store<T>'s
//!     drop_state:  unsafe fn(*mut u8)         // destructor for state
//! }
//! ```
//!
//! # Invariants
//!
//! * `store` is a valid, non-null pointer to a heap-allocated `Store<T>`.
//! * `type_id == TypeId::of::<T>()` where `T` is the store's scalar type.
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
//! `Node` construction is `unsafe` — the caller (typed `Scenario` methods)
//! must uphold the invariants above.  Once constructed, `Graph` methods
//! (`flush`, `dispatch`) are safe by relying on these invariants.

use std::any::TypeId;

use crate::operator::Operator;
use crate::store::Store;
use crate::types::{InputKinds, Scalar};

// ---------------------------------------------------------------------------
// Function pointer types
// ---------------------------------------------------------------------------

/// Type-erased compute function.
///
/// # Arguments
///
/// * `input_ptrs` — `&[*const u8]` pointing to input `Store<T>`'s.
/// * `output_store` — `*mut u8` pointing to the output `Store<T>`.
/// * `state` — `*mut u8` pointing to the operator's `State`.
/// * `timestamp` — flush timestamp for the new element.
///
/// # Returns
///
/// `true` if an output value was produced.
///
/// # Safety
///
/// All pointers must be valid for their respective types.  The output store
/// must not alias any input store.
pub(super) type ComputeFn = unsafe fn(&[*const u8], *mut u8, *mut u8, i64) -> bool;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// Type-erased DAG node: owns a [`Store`] and optionally a [`Closure`].
///
/// See [module-level docs](self) for layout and invariants.
pub(super) struct Node {
    /// Heap-allocated `Store<T>` (via `Box::into_raw`).
    pub store: *mut u8,
    /// `TypeId::of::<T>()` for the scalar type `T`.
    pub type_id: TypeId,
    /// Points into `Store<T>.shape` (a `Box<[usize]>` inside the
    /// heap-allocated Store).  Valid for the node's lifetime.
    shape: *const [usize],
    /// Operator closure, or `None` for source / bare nodes.
    pub closure: Option<Closure>,
    /// Downstream node indices (nodes whose closures read this node).
    pub edges: Vec<usize>,
    /// Drop the store: `drop(Box::from_raw(ptr as *mut Store<T>))`.
    drop_store: unsafe fn(*mut u8),
}

// SAFETY: Node owns the heap allocation behind `store`.  All access is
// single-threaded (Graph is not Sync).
unsafe impl Send for Node {}

impl Node {
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
    #[cfg(feature = "python")]
    pub fn stride(&self) -> usize {
        self.shape().iter().product::<usize>()
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // Drop the closure's state first (if any).
        if let Some(ref closure) = self.closure {
            // SAFETY: `state` was allocated by `Box::into_raw` in `new_closure`.
            unsafe { (closure.drop_state)(closure.state) };
        }
        // Drop the store (which also frees the shape that `self.shape` points to).
        // SAFETY: `store` was allocated by `Box::into_raw` in `new_node`.
        unsafe { (self.drop_store)(self.store) };
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
    pub compute_fn: ComputeFn,
    /// Heap-allocated operator state (`Box::into_raw(Box::new(op.init()))`).
    pub state: *mut u8,
    /// Pre-collected pointers to input `Store<T>`'s.
    pub input_ptrs: Box<[*const u8]>,
    /// Drop the state: `drop(Box::from_raw(ptr as *mut State))`.
    pub(super) drop_state: unsafe fn(*mut u8),
}

// SAFETY: Closure fields are only accessed from a single thread.
unsafe impl Send for Closure {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Drop a heap-allocated `T`.
///
/// # Safety
///
/// `ptr` must have been created by `Box::into_raw(Box::new(..))` for type `T`.
unsafe fn drop_box<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}

/// Type-erased compute entry point, monomorphised per operator type.
///
/// Pushes the store's default values as a new element, then calls
/// `Op::compute` to overwrite it.  On failure (returns `false`),
/// the element is popped (rollback).
///
/// # Safety
///
/// * Each `input_ptrs[i]` must point to a valid `Store` of the type expected
///   by `Op::Inputs` at position `i`.
/// * `output_ptr` must point to a valid `Store<Op::Scalar>`.
/// * `state_ptr` must point to a valid `Op::State`.
/// * `output_ptr` must not alias any `input_ptrs[i]`.
unsafe fn erased_compute<Op: Operator>(
    input_ptrs: &[*const u8],
    output_ptr: *mut u8,
    state_ptr: *mut u8,
    timestamp: i64,
) -> bool {
    unsafe {
        let store = &mut *(output_ptr as *mut Store<Op::Scalar>);
        let state = &mut *(state_ptr as *mut Op::State);
        store.push_default(timestamp);
        let output = store.current_view_mut();
        let inputs = <Op::Inputs as InputKinds>::from_ptrs(input_ptrs);
        let produced = Op::compute(state, inputs, output);
        if produced {
            store.commit();
        } else {
            store.rollback();
        }
        produced
    }
}

// ---------------------------------------------------------------------------
// Typed construction helpers (called by Scenario)
// ---------------------------------------------------------------------------

/// Create a new bare node (no closure) for a given scalar type.
///
/// The store is heap-allocated; the returned `Node` owns it.
pub(super) fn new_node<T: Scalar>(store: Store<T>) -> Node {
    let store_ptr = Box::into_raw(Box::new(store));
    // SAFETY: store_ptr is valid; Store<T>.shape is a Box<[usize]> whose
    // heap allocation is stable for the Store's lifetime.
    let shape_ptr = unsafe { (*store_ptr).shape() as *const [usize] };
    Node {
        store: store_ptr as *mut u8,
        type_id: TypeId::of::<T>(),
        shape: shape_ptr,
        closure: None,
        edges: Vec::new(),
        drop_store: drop_box::<Store<T>>,
    }
}

/// Build a [`Closure`] for an operator.
///
/// # Arguments
///
/// * `input_ptrs` — pointers to input stores, collected at registration.
/// * `op_state` — the operator's initial state (from [`Operator::init`]).
pub(super) fn new_closure<Op: Operator>(
    input_ptrs: Box<[*const u8]>,
    op_state: Op::State,
) -> Closure {
    Closure {
        compute_fn: erased_compute::<Op>,
        state: Box::into_raw(Box::new(op_state)) as *mut u8,
        input_ptrs,
        drop_state: drop_box::<Op::State>,
    }
}
