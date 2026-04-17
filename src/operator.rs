//! Operator trait and type-erased operator.
//!
//! # Public items
//!
//! - [`Operator`] — synchronous computation trait.  Two hierarchical views
//!   are built from flat graph buffers and passed to
//!   [`compute`](Operator::compute): `inputs` (nested references via
//!   [`FlatRead`]) and `produced` (nested bits via [`BitRead`]).  The two
//!   are structurally parallel — same tree shape, different leaf types.
//! - [`ErasedOperator`] — type-erased operator combining init/compute/drop
//!   function pointers and `TypeId`s for runtime validation.
//! - [`InitFn`] / [`ComputeFn`] — type aliases for the erased signatures.

use std::any::TypeId;

use super::data::Instant;
use super::data::{BitRead, FlatRead, FlatWrite, InputTypes};

/// A synchronous computation node that reads typed inputs and writes a typed
/// output.
pub trait Operator: 'static {
    /// Mutable runtime state.
    type State: Send + 'static;
    /// Input tree (e.g. `(Input<Array<f64>>, Input<Array<f64>>)`).
    type Inputs: InputTypes + ?Sized;
    /// Output type.
    type Output: Send + 'static;

    /// Consume the spec and produce initial state and output.
    fn init(
        self,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        timestamp: Instant,
    ) -> (Self::State, Self::Output);

    /// Update the output from inputs and current state.
    ///
    /// `produced` mirrors `inputs` in shape: each leaf is a `bool` flagging
    /// whether that input produced in this flush cycle.  Slice branches
    /// expose a lazy [`SliceProduced`](crate::data::SliceProduced) view —
    /// bits are only read when the operator descends into elements.
    ///
    /// Returns `true` if downstream propagation should occur.
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: Instant,
        produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool;
}

/// Type-erased initialization closure.
///
/// Receives the flat input pointer buffer and timestamp; returns
/// `(state_ptr, output_ptr)` from `Box::into_raw`.
pub type InitFn = Box<dyn FnOnce(&[*const u8], Instant) -> (*mut u8, *mut u8)>;

/// Type-erased compute function pointer.
///
/// The flat-buffer arguments `produced_words` / `produced_bit_off` /
/// `produced_num_inputs` describe a bit range covering this operator's
/// inputs; the monomorphised body threads them through a [`BitRead`] cursor
/// to build the operator's concrete `Produced<'_>` tree.
pub type ComputeFn = unsafe fn(
    state: *mut u8,
    input_ptrs: &[*const u8],
    output: *mut u8,
    timestamp: Instant,
    produced_words: &[u64],
    produced_bit_off: usize,
    produced_num_inputs: usize,
) -> bool;

/// Type-erased representation of an operator.
pub struct ErasedOperator {
    state_type_id: TypeId,
    input_type_ids: Box<[TypeId]>,
    output_type_id: TypeId,
    init_fn: InitFn,
    compute_fn: ComputeFn,
    state_drop_fn: unsafe fn(*mut u8),
    output_drop_fn: unsafe fn(*mut u8),
}

impl ErasedOperator {
    /// Construct from raw type-erased components.
    ///
    /// # Safety
    ///
    /// All function pointers must correctly interpret their pointer arguments
    /// as the types encoded in the corresponding `TypeId` fields.
    pub unsafe fn new(
        state_type_id: TypeId,
        input_type_ids: Box<[TypeId]>,
        output_type_id: TypeId,
        init_fn: InitFn,
        compute_fn: ComputeFn,
        state_drop_fn: unsafe fn(*mut u8),
        output_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            state_type_id,
            input_type_ids,
            output_type_id,
            init_fn,
            compute_fn,
            state_drop_fn,
            output_drop_fn,
        }
    }

    /// Construct from a typed [`Operator`] whose `Inputs` is `Sized`.
    pub fn from_operator<O: Operator>(op: O) -> Self
    where
        O::Inputs: Sized,
    {
        let arity = O::Inputs::arity();
        let mut type_ids = vec![TypeId::of::<()>(); arity];
        {
            let mut writer = FlatWrite::new(&mut type_ids);
            O::Inputs::type_ids_to_flat(&mut writer);
        }
        Self::from_operator_with_type_ids(op, type_ids.into_boxed_slice())
    }

    /// Construct from a typed [`Operator`] with an externally-provided
    /// type-id list.
    ///
    /// Used for operators with `!Sized` `Inputs` (e.g. `[Input<T>]`), where
    /// the element count and per-element `TypeId`s are derived from the
    /// handles rather than from the type alone.
    pub fn from_operator_with_type_ids<O: Operator>(op: O, input_type_ids: Box<[TypeId]>) -> Self {
        Self {
            state_type_id: TypeId::of::<O::State>(),
            input_type_ids,
            output_type_id: TypeId::of::<O::Output>(),
            init_fn: Box::new(move |input_ptrs: &[*const u8], timestamp: Instant| {
                let mut reader = FlatRead::new(input_ptrs);
                let inputs = unsafe { O::Inputs::refs_from_flat(&mut reader) };
                let (state, output) = op.init(inputs, timestamp);
                (
                    Box::into_raw(Box::new(state)) as *mut u8,
                    Box::into_raw(Box::new(output)) as *mut u8,
                )
            }),
            compute_fn: erased_compute_fn::<O>,
            state_drop_fn: erased_drop_fn::<O::State>,
            output_drop_fn: erased_drop_fn::<O::Output>,
        }
    }

    pub fn state_type_id(&self) -> TypeId {
        self.state_type_id
    }

    pub fn input_type_ids(&self) -> &[TypeId] {
        &self.input_type_ids
    }

    pub fn output_type_id(&self) -> TypeId {
        self.output_type_id
    }

    pub fn compute_fn(&self) -> ComputeFn {
        self.compute_fn
    }

    pub fn state_drop_fn(&self) -> unsafe fn(*mut u8) {
        self.state_drop_fn
    }

    pub fn output_drop_fn(&self) -> unsafe fn(*mut u8) {
        self.output_drop_fn
    }

    /// Consume the init closure, producing `(state_ptr, output_ptr)`.
    ///
    /// # Safety
    ///
    /// `input_ptrs` must point to valid objects matching `input_type_ids`.
    pub unsafe fn init(self, input_ptrs: &[*const u8], timestamp: Instant) -> (*mut u8, *mut u8) {
        (self.init_fn)(input_ptrs, timestamp)
    }
}

/// Type-erased compute function, monomorphised per operator type.
///
/// Parallel tree construction: [`FlatRead`] over `input_ptrs` feeds
/// [`refs_from_flat`](InputTypes::refs_from_flat); [`BitRead`] over
/// `produced_words` feeds
/// [`produced_from_flat`](InputTypes::produced_from_flat).
unsafe fn erased_compute_fn<O: Operator>(
    state_ptr: *mut u8,
    input_ptrs: &[*const u8],
    output_ptr: *mut u8,
    timestamp: Instant,
    produced_words: &[u64],
    produced_bit_off: usize,
    produced_num_inputs: usize,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut O::State) };
    let mut ptr_reader = FlatRead::new(input_ptrs);
    let inputs = unsafe { O::Inputs::refs_from_flat(&mut ptr_reader) };
    let mut bit_reader =
        BitRead::from_parts(produced_words, produced_bit_off, produced_num_inputs);
    let produced = O::Inputs::produced_from_flat(&mut bit_reader);
    let output = unsafe { &mut *(output_ptr as *mut O::Output) };
    O::compute(state, inputs, output, timestamp, produced)
}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
