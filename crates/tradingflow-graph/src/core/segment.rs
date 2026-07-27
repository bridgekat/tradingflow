use std::any::TypeId;

use super::cell::ErasedCell;

/// Type-erased compute function.
///
/// Before this call, every pointer in `input_ptrs` must point to a value of
/// the type at the corresponding location in `input_types`. The pointers in
/// `output_ptrs` may be null or invalid and must be overwritten by this call.
///
/// After this call, every pointer in `output_ptrs` must point to a value of
/// the type at the corresponding location in `output_types`, which must remain
/// completely valid as long as both inputs and `state` remain unchanged.
///
/// The `context` and `state` point to the graph-level context and the
/// node-level internal state, respectively.
pub type ComputeFn = unsafe fn(
    input_ptrs: *const [*const ()],
    output_ptrs: *mut [*const ()],
    state: *mut (),
    context: *const (),
);

/// Type-erased reset function.
///
/// Safety contract is the same as [`ComputeFn`].
pub type ResetFn = unsafe fn(
    input_ptrs: *const [*const ()],
    output_ptrs: *mut [*const ()],
    state: *mut (),
    context: *const (),
);

/// Type-erased segment definition.
#[derive(Debug)]
pub struct Segment {
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
    /// Initial output value pointers. Can reference into the state.
    output_ptrs: Box<[*const ()]>,
}

impl Segment {
    /// # Safety
    ///
    /// The given `compute_fn` and `reset_fn` must correctly handle the
    /// provided `input_types` and `state`.
    pub unsafe fn new(
        input_types: Box<[TypeId]>,
        output_types: Box<[TypeId]>,
        compute_fn: ComputeFn,
        reset_fn: ResetFn,
        state: ErasedCell,
        output_ptrs: Box<[*const ()]>,
    ) -> Self {
        Self {
            input_types,
            output_types,
            compute_fn,
            reset_fn,
            state,
            output_ptrs,
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
    ) {
        (
            self.input_types,
            self.output_types,
            self.compute_fn,
            self.reset_fn,
            self.state,
            self.output_ptrs,
        )
    }
}
