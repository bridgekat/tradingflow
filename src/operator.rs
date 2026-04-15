//! Operator trait, type-erased operator, and notification context.
//!
//! This module defines the [`Operator`] trait for synchronous computation
//! nodes, the [`Notify`] context that reports which inputs produced new
//! output during a flush cycle, and the [`ErasedOperator`] wrapper for
//! type-erased DAG dispatch.
//!
//! # Public items
//!
//! - [`Notify`] — notification context for [`Operator::compute`].
//! - [`Operator`] — synchronous computation trait with associated state,
//!   input, and output types.
//! - [`ErasedOperator`] — type-erased operator combining init/compute/drop
//!   function pointers, an erased runtime shape, and `TypeId`s for runtime
//!   type checking.
//! - [`InitFn`] / [`ComputeFn`] — type aliases for the erased function
//!   pointer signatures.

use std::any::{Any, TypeId};

use super::data::Instant;
use super::data::{FlatRead, FlatShapeFromArity, FlatWrite, InputTypes, Notify};

/// A synchronous computation node that reads typed inputs and writes a
/// typed output.
///
/// # Lifecycle
///
/// 1. [`init`](Self::init) consumes the spec, producing runtime
///    [`State`](Self::State) and the initial [`Output`](Self::Output).
/// 2. [`compute`](Self::compute) is called on each flush to update the
///    output.
pub trait Operator: 'static {
    /// Mutable runtime state.
    type State: Send + 'static;
    /// Input tree description (e.g. `(Input<Array<f64>>, Input<Array<f64>>)`).
    type Inputs: InputTypes + ?Sized;
    /// Output type.
    type Output: Send + 'static;

    /// Whether this operator can be gated by a clock trigger.
    ///
    /// In general, operators assuming time-series semantics return `true`,
    /// while those assuming message-passing semantics return `false`
    /// so that messages are not accidentally dropped.
    ///
    /// The default is `true`.
    fn is_clock_triggerable(&self) -> bool {
        true
    }

    /// Consume the spec and produce initial state and output.
    fn init(
        self,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        timestamp: Instant,
    ) -> (Self::State, Self::Output);

    /// Update the output from inputs and current state.
    ///
    /// The [`Notify`] context reports which inputs produced new output
    /// in the current flush cycle via [`Notify::produced`] (list of
    /// positions) and [`Notify::input_produced`] (per-position booleans).
    ///
    /// Returns `true` if downstream propagation should occur.
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: Instant,
        notify: &Notify<'_>,
    ) -> bool;
}

/// Type-erased initialization closure for an operator.
///
/// # Parameters
///
/// * `input_ptrs: &[*const u8]` — pointers to input node values (flat).
/// * `shape: &(dyn Any + Send)` — erased [`InputTypes::Shape`]; the
///   monomorphized body downcasts to the concrete type to build `Refs`.
/// * `timestamp: Instant` — initial timestamp.
///
/// # Returns
///
/// * `state_ptr: *mut u8` — from [`Box::into_raw`], points to `O::State`.
/// * `output_ptr: *mut u8` — from [`Box::into_raw`], points to `O::Output`.
pub type InitFn =
    Box<dyn FnOnce(&[*const u8], &(dyn Any + Send), Instant) -> (*mut u8, *mut u8)>;

/// Type-erased compute function pointer for an operator.
///
/// # Parameters
///
/// * `state_ptr: *mut u8` — points to `O::State`.
/// * `input_ptrs: &[*const u8]` — pointers to input node values (flat).
/// * `output_ptr: *mut u8` — points to `O::Output`.
/// * `timestamp: Instant` — current flush timestamp.
/// * `notify: &Notify` — notification context.
/// * `shape: &(dyn Any + Send)` — erased [`InputTypes::Shape`]; the
///   monomorphized body downcasts to the concrete type to build `Refs`.
///
/// # Returns
///
/// * `true` if downstream propagation should occur.
pub type ComputeFn =
    unsafe fn(*mut u8, &[*const u8], *mut u8, Instant, &Notify, &(dyn Any + Send)) -> bool;

/// Type-erased representation of an operator.
///
/// # Lifecycle
///
/// 1. Created via [`from_operator`](ErasedOperator::from_operator) (safe,
///    typed) or [`new`](ErasedOperator::new) (`unsafe`, raw).
/// 2. Consumed by [`Scenario::add_erased_operator`], which validates input
///    types, calls [`init`](ErasedOperator::init), and constructs the DAG node.
///    `init` returns the runtime shape (erased) so it can be stored in
///    [`OperatorState`](crate::scenario::node::OperatorState) for later
///    compute calls.
pub struct ErasedOperator {
    state_type_id: TypeId,
    input_type_ids: Box<[TypeId]>,
    output_type_id: TypeId,
    is_clock_triggerable: bool,
    /// Erased runtime shape describing the operator's input tree.  Used by
    /// `init_fn` and [`compute_fn`](Self::compute_fn) to build nested
    /// `Refs` from flat input pointers.
    shape: Box<dyn Any + Send>,
    init_fn: InitFn,
    compute_fn: ComputeFn,
    state_drop_fn: unsafe fn(*mut u8),
    output_drop_fn: unsafe fn(*mut u8),
}

impl ErasedOperator {
    /// Construct from raw, type-erased components.
    ///
    /// # Safety
    ///
    /// * `shape`'s concrete type must match `<O::Inputs as InputTypes>::Shape`
    ///   for the operator type `O` implied by `init_fn` and `compute_fn`.
    /// * `init_fn` must return valid `(state_ptr, output_ptr)` from
    ///   [`Box::into_raw`] pointing to objects of types `state_type_id` and
    ///   `output_type_id` respectively.
    /// * `compute_fn` must correctly interpret `state_ptr`, `input_ptrs`,
    ///   and `output_ptr` as pointers to objects of types `state_type_id`,
    ///   `input_type_ids`, and `output_type_id` respectively.
    /// * `state_drop_fn` must correctly drop [`Box::from_raw`] pointing to
    ///   an object of type `state_type_id`.
    /// * `output_drop_fn` must correctly drop [`Box::from_raw`] pointing to
    ///   an object of type `output_type_id`.
    pub unsafe fn new(
        state_type_id: TypeId,
        input_type_ids: Box<[TypeId]>,
        output_type_id: TypeId,
        is_clock_triggerable: bool,
        shape: Box<dyn Any + Send>,
        init_fn: InitFn,
        compute_fn: ComputeFn,
        state_drop_fn: unsafe fn(*mut u8),
        output_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            input_type_ids,
            output_type_id,
            state_type_id,
            is_clock_triggerable,
            shape,
            init_fn,
            compute_fn,
            state_drop_fn,
            output_drop_fn,
        }
    }

    /// Construct from a typed [`Operator`] using a flat shape built from
    /// just the arity.
    ///
    /// Convenience wrapper around [`from_operator`](Self::from_operator)
    /// for operators whose [`Inputs`](Operator::Inputs) has a flat shape
    /// (tuple of [`Input<T>`] or [`Slice<Input<T>>`] — anything that
    /// implements [`FlatShapeFromArity`]).  Used by the Python bridge and
    /// similar type-erased registration paths where the concrete shape
    /// structure is not available at compile time.
    pub fn from_flat_operator<O: Operator>(op: O, arity: usize) -> Self
    where
        <O::Inputs as InputTypes>::Shape: FlatShapeFromArity,
    {
        let shape = <<O::Inputs as InputTypes>::Shape as FlatShapeFromArity>::flat_shape_from_arity(
            arity,
        );
        Self::from_operator(op, shape)
    }

    /// Construct from a typed [`Operator`] plus its runtime input shape.
    pub fn from_operator<O: Operator>(op: O, shape: <O::Inputs as InputTypes>::Shape) -> Self {
        let is_clock_triggerable = op.is_clock_triggerable();
        let arity = <O::Inputs as InputTypes>::arity(&shape);
        let mut type_ids: Vec<TypeId> = vec![TypeId::of::<()>(); arity];
        {
            let mut writer = FlatWrite::new(&mut type_ids);
            <O::Inputs as InputTypes>::write_type_ids(&shape, &mut writer);
        }
        let shape_box: Box<dyn Any + Send> = Box::new(shape);

        Self {
            state_type_id: TypeId::of::<O::State>(),
            input_type_ids: type_ids.into_boxed_slice(),
            output_type_id: TypeId::of::<O::Output>(),
            is_clock_triggerable,
            shape: shape_box,
            init_fn: Box::new(
                move |input_ptrs: &[*const u8],
                      shape: &(dyn Any + Send),
                      timestamp: Instant| {
                    let concrete = shape
                        .downcast_ref::<<O::Inputs as InputTypes>::Shape>()
                        .expect("shape type mismatch in init_fn");
                    // SAFETY: call site guarantees `input_ptrs` point to valid
                    // objects of types matching `input_type_ids`, and the shape
                    // corresponds to O::Inputs (verified by downcast).
                    let mut reader = FlatRead::new(input_ptrs);
                    let inputs = unsafe {
                        <O::Inputs as InputTypes>::refs_from_flat(&mut reader, concrete)
                    };
                    let (state, output) = op.init(inputs, timestamp);
                    let state = Box::into_raw(Box::new(state)) as *mut u8;
                    let output = Box::into_raw(Box::new(output)) as *mut u8;
                    (state, output)
                },
            ),
            compute_fn: erased_compute_fn::<O>,
            state_drop_fn: erased_drop_fn::<O::State>,
            output_drop_fn: erased_drop_fn::<O::Output>,
        }
    }

    /// Whether this operator can be gated by a clock trigger.
    pub fn is_clock_triggerable(&self) -> bool {
        self.is_clock_triggerable
    }

    /// The [`TypeId`] of the operator's state type.
    pub fn state_type_id(&self) -> TypeId {
        self.state_type_id
    }

    /// The [`TypeId`]s of the operator's input types, one per flat input
    /// position.
    pub fn input_type_ids(&self) -> &[TypeId] {
        &self.input_type_ids
    }

    /// The [`TypeId`] of the operator's output type.
    pub fn output_type_id(&self) -> TypeId {
        self.output_type_id
    }

    /// The type-erased compute function pointer.
    pub fn compute_fn(&self) -> ComputeFn {
        self.compute_fn
    }

    /// The type-erased drop function for the operator's state.
    pub fn state_drop_fn(&self) -> unsafe fn(*mut u8) {
        self.state_drop_fn
    }

    /// The type-erased drop function for the operator's output.
    pub fn output_drop_fn(&self) -> unsafe fn(*mut u8) {
        self.output_drop_fn
    }

    /// Consume the init closure, producing `(state_ptr, output_ptr, shape)`.
    ///
    /// The runtime shape is moved out of `self` and returned alongside the
    /// state and output pointers so the caller can stash it in
    /// [`OperatorState`](crate::scenario::node::OperatorState) for later
    /// compute calls.
    ///
    /// # Safety
    ///
    /// * `input_ptrs` must point to valid objects whose types match with
    ///   [`self.input_type_ids`](Self::input_type_ids).
    pub unsafe fn init(
        self,
        input_ptrs: &[*const u8],
        timestamp: Instant,
    ) -> (*mut u8, *mut u8, Box<dyn Any + Send>) {
        let Self {
            shape, init_fn, ..
        } = self;
        let (state_ptr, output_ptr) = init_fn(input_ptrs, &*shape, timestamp);
        (state_ptr, output_ptr, shape)
    }
}

/// Type-erased compute function, monomorphised per operator type.
unsafe fn erased_compute_fn<O: Operator>(
    state_ptr: *mut u8,
    input_ptrs: &[*const u8],
    output_ptr: *mut u8,
    timestamp: Instant,
    notify: &Notify<'_>,
    shape: &(dyn Any + Send),
) -> bool {
    let concrete = shape
        .downcast_ref::<<O::Inputs as InputTypes>::Shape>()
        .expect("shape type mismatch in erased_compute_fn");
    let state = unsafe { &mut *(state_ptr as *mut O::State) };
    let mut reader = FlatRead::new(input_ptrs);
    let inputs = unsafe { <O::Inputs as InputTypes>::refs_from_flat(&mut reader, concrete) };
    let output = unsafe { &mut *(output_ptr as *mut O::Output) };
    O::compute(state, inputs, output, timestamp, notify)
}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
