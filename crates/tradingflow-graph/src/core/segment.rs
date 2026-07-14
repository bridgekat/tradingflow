use std::any::TypeId;

use super::cell::ErasedCell;

/// Type-erased compute function.
///
/// Before this call, every pointer in `in_ptrs` must point to a value of the
/// type at the corresponding location in `input_types`. The pointers in
/// `out_ptrs` may be null or invalid and must be overwritten by this call.
///
/// After this call, every pointer in `out_ptrs` must point to a value of the
/// type at the corresponding location in `output_types`, which must remain
/// completely valid as long as both inputs and `state` remain unchanged.
///
/// The `context` and `state` point to the graph-level context and the
/// node-level internal state, respectively.
pub type ComputeFn = unsafe fn(
    in_flags: *const [bool],
    in_ptrs: *const [*const ()],
    out_flags: *mut [bool],
    out_ptrs: *mut [*const ()],
    context: *const (),
    state: *mut (),
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
    /// Initial state.
    state: ErasedCell,
    /// Initial output notify flags.
    output_flags: Box<[bool]>,
    /// Initial output value pointers, may reference into states.
    output_ptrs: Box<[*const ()]>,
}

impl Segment {
    /// # Safety
    ///
    /// The given `compute_fn` must correctly handle the provided `input_types`,
    /// and `state`.
    pub unsafe fn new(
        input_types: Box<[TypeId]>,
        output_types: Box<[TypeId]>,
        compute_fn: ComputeFn,
        state: ErasedCell,
        output_flags: Box<[bool]>,
        output_ptrs: Box<[*const ()]>,
    ) -> Self {
        Self {
            input_types,
            output_types,
            compute_fn,
            state,
            output_flags,
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
        ErasedCell,
        Box<[bool]>,
        Box<[*const ()]>,
    ) {
        (
            self.input_types,
            self.output_types,
            self.compute_fn,
            self.state,
            self.output_flags,
            self.output_ptrs,
        )
    }
}
