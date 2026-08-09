use std::any::TypeId;

use super::cell::ErasedCell;

/// Type-erased compute function.
///
/// Before this call, every pointer in `input_ptrs` must point to a value of
/// the corresponding type specified in `input_types`.
///
/// After this call, every output slot in `state` (addressed by `output_ptrs`)
/// must hold a value of the corresponding type specified in `output_types`
/// that is safe to access across threads.
///
/// The `context` and `state` point to the graph-level context and the
/// node-level internal state, respectively.
pub type ComputeFn = unsafe fn(input_ptrs: *const [*const ()], state: *mut (), context: *const ());

/// Type-erased reset function.
///
/// Safety contract is the same as [`ComputeFn`].
pub type ResetFn = unsafe fn(input_ptrs: *const [*const ()], state: *mut (), context: *const ());

/// Type-erased node definition.
#[derive(Debug)]
pub struct Node {
    /// Expected input types, for wiring checks.
    input_types: Box<[TypeId]>,
    /// Declared outputs types, for wiring checks.
    output_types: Box<[TypeId]>,
    /// The compute function.
    compute_fn: ComputeFn,
    /// The reset function.
    reset_fn: ResetFn,
    /// Initial state.
    state: ErasedCell,
    /// Output slot pointers. Reference stable addresses in `state`.
    output_ptrs: Box<[*const ()]>,
    /// Whether the node is expensive enough to be worth a parallel task.
    is_heavy: bool,
}

impl Node {
    /// # Safety
    ///
    /// The given `compute_fn` and `reset_fn` must correctly handle the
    /// provided `input_types` and `state` (see [`ComputeFn`]).
    ///
    /// Every pointer in `output_ptrs` must target stable storage reachable
    /// from `state`, which must hold an initialized value of the corresponding
    /// type specified in `output_types` that is safe to access across threads.
    pub unsafe fn new(
        input_types: Box<[TypeId]>,
        output_types: Box<[TypeId]>,
        compute_fn: ComputeFn,
        reset_fn: ResetFn,
        state: ErasedCell,
        output_ptrs: Box<[*const ()]>,
        is_heavy: bool,
    ) -> Self {
        Self {
            input_types,
            output_types,
            compute_fn,
            reset_fn,
            state,
            output_ptrs,
            is_heavy,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        Box<[TypeId]>,
        Box<[TypeId]>,
        ComputeFn,
        ResetFn,
        ErasedCell,
        Box<[*const ()]>,
        bool,
    ) {
        (
            self.input_types,
            self.output_types,
            self.compute_fn,
            self.reset_fn,
            self.state,
            self.output_ptrs,
            self.is_heavy,
        )
    }
}
