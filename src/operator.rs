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
//!   function pointers and `TypeId`s for runtime type checking.
//! - [`InitFn`] / [`ComputeFn`] — type aliases for the erased function
//!   pointer signatures.

use std::any::TypeId;

use super::types::{InputTypes, Notify};

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
    /// Input types (e.g. `(Array<f64>, Array<f64>)`).
    type Inputs: InputTypes + ?Sized;
    /// Output type.
    type Output: Send + 'static;

    /// Whether this operator can be gated by a clock trigger.
    ///
    /// Operators that use message-passing semantics (relying on
    /// [`Notify::produced_inputs`] to track which inputs produced)
    /// should return `false` — they must be triggered by their data
    /// inputs so they never miss a message.
    ///
    /// The default is `true` (clock-triggerable).
    fn is_clock_triggerable(&self) -> bool {
        true
    }

    /// Consume the spec and produce initial state and output.
    fn init(
        self,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        timestamp: i64,
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
        timestamp: i64,
        notify: &Notify<'_>,
    ) -> bool;
}

/// Type-erased initialization closure for an operator.
///
/// # Parameters
///
/// * `input_ptrs: &[*const u8]` — pointers to input node values.
/// * `timestamp: i64` — initial timestamp.
///
/// # Returns
///
/// * `state_ptr: *mut u8` — from [`Box::into_raw`], points to `O::State`.
/// * `output_ptr: *mut u8` — from [`Box::into_raw`], points to `O::Output`.
pub type InitFn = Box<dyn FnOnce(&[*const u8], i64) -> (*mut u8, *mut u8)>;

/// Type-erased compute function pointer for an operator.
///
/// # Parameters
///
/// * `state_ptr: *mut u8` — points to `O::State`.
/// * `input_ptrs: &[*const u8]` — pointers to input node values.
/// * `output_ptr: *mut u8` — points to `O::Output`.
/// * `timestamp: i64` — current flush timestamp.
/// * `notify: &Notify` — notification context.
///
/// # Returns
///
/// * `true` if downstream propagation should occur.
pub type ComputeFn = unsafe fn(*mut u8, &[*const u8], *mut u8, i64, &Notify) -> bool;

/// Type-erased representation of an operator.
///
/// # Lifecycle
///
/// 1. Created via [`from_operator`](ErasedOperator::from_operator) (safe,
///    typed) or [`new`](ErasedOperator::new) (`unsafe`, raw).
/// 2. Consumed by [`Scenario::add_erased_operator`], which validates input
///    types, calls [`init`](ErasedOperator::init), and constructs the DAG node.
pub struct ErasedOperator {
    state_type_id: TypeId,
    input_type_ids: Box<[TypeId]>,
    output_type_id: TypeId,
    is_clock_triggerable: bool,
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
            init_fn,
            compute_fn,
            state_drop_fn,
            output_drop_fn,
        }
    }

    /// Construct from a typed [`Operator`].
    pub fn from_operator<O: Operator>(op: O, arity: usize) -> Self {
        let is_clock_triggerable = op.is_clock_triggerable();
        Self {
            state_type_id: TypeId::of::<O::State>(),
            input_type_ids: <O::Inputs as InputTypes>::type_ids(arity),
            output_type_id: TypeId::of::<O::Output>(),
            is_clock_triggerable,
            init_fn: Box::new(move |input_ptrs: &[*const u8], timestamp: i64| {
                // SAFETY: call site guarantees `input_ptrs` point to valid objects of types
                // matching `input_type_ids`.
                let inputs = unsafe { <O::Inputs as InputTypes>::from_ptrs(input_ptrs) };
                let (state, output) = op.init(inputs, timestamp);
                let state = Box::into_raw(Box::new(state)) as *mut u8;
                let output = Box::into_raw(Box::new(output)) as *mut u8;
                (state, output)
            }),
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

    /// The [`TypeId`]s of the operator's input types, one per input position.
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

    /// Consume the init closure, producing `(state_ptr, output_ptr)`.
    ///
    /// # Safety
    ///
    /// * `input_ptrs` must point to valid objects whose types match with
    ///   [`self.input_type_ids`](Self::input_type_ids).
    pub unsafe fn init(self, input_ptrs: &[*const u8], timestamp: i64) -> (*mut u8, *mut u8) {
        (self.init_fn)(input_ptrs, timestamp)
    }
}

/// Type-erased compute function, monomorphised per operator type.
unsafe fn erased_compute_fn<O: Operator>(
    state_ptr: *mut u8,
    input_ptrs: &[*const u8],
    output_ptr: *mut u8,
    timestamp: i64,
    notify: &Notify<'_>,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut O::State) };
    let inputs = unsafe { <O::Inputs as InputTypes>::from_ptrs(input_ptrs) };
    let output = unsafe { &mut *(output_ptr as *mut O::Output) };
    O::compute(state, inputs, output, timestamp, notify)
}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
