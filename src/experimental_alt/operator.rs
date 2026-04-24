//! Operator trait and type-erased wrapper for the wavefront runtime.
//!
//! # Relation to `crate::operator::Operator`
//!
//! The new [`Operator`] trait has the **same method signatures** as the
//! existing one — existing operator bodies port over verbatim — with one
//! addition: `Output: Clone`.  The scheduler clones the init-returned
//! template per timestamp to give `compute` a fresh buffer to write into,
//! which the auto-managed output queue then takes ownership of.  In the
//! legacy serial runtime the output was an in-place-mutated single buffer;
//! under cross-`t` pipelining a writer at `t+1` would otherwise clobber a
//! reader at `t`.
//!
//! [`NodeKind`] classifies nodes at registration time into `Source` or
//! `Operator`.  A "Const" — 0-input operator — is treated as an
//! [`NodeKind::Operator`] whose compute is never called (it has no trigger
//! edges pointing in); its initial output sits in the queue indefinitely
//! and is what downstream reads pick up.

use std::any::TypeId;

use super::data::{BitRead, FlatRead, FlatWrite, InputTypes, Instant};

// ===========================================================================
// Operator trait
// ===========================================================================

/// A synchronous computation node that reads typed inputs and writes a
/// typed output.
///
/// Under the wavefront runtime the `output` parameter of [`compute`] is a
/// freshly-allocated buffer (cloned from the init template) — the operator
/// writes into it, and if `compute` returns `true` the scheduler takes
/// ownership of the filled buffer and commits it to the node's output
/// queue.  Returning `false` discards the buffer and suppresses downstream
/// propagation for this timestamp.
pub trait Operator: Send + 'static {
    /// Mutable runtime state.
    type State: Send + 'static;
    /// Input tree (e.g. `(Input<Array<f64>>, Input<Array<f64>>)`).
    type Inputs: InputTypes + ?Sized;
    /// Output type.  `Clone` is required so the scheduler can allocate a
    /// fresh per-timestamp buffer from the init-returned template.
    type Output: Clone + Send + 'static;

    /// Consume the spec and produce initial state and the output template.
    ///
    /// The template is cloned per timestamp to give `compute` a fresh
    /// buffer; see trait-level docs.
    fn init(
        self,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        timestamp: Instant,
    ) -> (Self::State, Self::Output);

    /// Compute the output from inputs and current state.
    ///
    /// `produced` mirrors `inputs` in shape: each leaf is a `bool` flagging
    /// whether that input fired this cycle.  Returns `true` if the written
    /// output should be committed and propagated downstream.
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: Instant,
        produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool;
}

// ===========================================================================
// NodeKind (runtime classification, not a trait-level concept)
// ===========================================================================

/// Role of a node within the computation graph.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum NodeKind {
    /// Driven by the ingest loop; emits slots at event timestamps.
    Source,
    /// Driven by trigger edges from upstream nodes.  A 0-input operator
    /// (Const) is an `Operator` with no incoming edges — never scheduled.
    Operator,
}

// ===========================================================================
// Type-erased wrappers
// ===========================================================================

/// Type-erased init closure producing `(state_ptr, output_template_ptr)`.
///
/// Both pointers come from `Box::into_raw`.
pub type InitFn = Box<dyn FnOnce(&[*const u8], Instant) -> (*mut u8, *mut u8) + Send>;

/// Type-erased compute function.
///
/// Writes into `output`, a `Box<Output>` freshly cloned from the template.
/// Returns `true` if downstream propagation should occur.
pub type ComputeFn = unsafe fn(
    state: *mut u8,
    input_ptrs: &[*const u8],
    output: *mut u8,
    timestamp: Instant,
    produced_words: &[u64],
    produced_bit_off: usize,
    produced_num_inputs: usize,
) -> bool;

/// Type-erased clone function for the output template.
pub type CloneFn = unsafe fn(*const u8) -> *mut u8;

/// Type-erased drop function for a single boxed value.
pub type DropFn = unsafe fn(*mut u8);

/// Type-erased representation of an [`Operator`].
///
/// Mirrors [`crate::operator::ErasedOperator`] with the addition of
/// [`clone_fn`](Self::clone_fn) (the scheduler uses it to mint fresh output
/// buffers per timestamp).
pub struct ErasedOperator {
    state_type_id: TypeId,
    input_type_ids: Box<[TypeId]>,
    output_type_id: TypeId,
    init_fn: InitFn,
    compute_fn: ComputeFn,
    clone_fn: CloneFn,
    state_drop_fn: DropFn,
    output_drop_fn: DropFn,
}

impl ErasedOperator {
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
    /// Used for operators whose `Inputs` is `!Sized` (e.g.
    /// `[Input<Array<T>>]`) — the element count and per-element `TypeId`s
    /// come from the handles rather than from the type alone.
    pub fn from_operator_with_type_ids<O: Operator>(
        op: O,
        input_type_ids: Box<[TypeId]>,
    ) -> Self {
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
            clone_fn: erased_clone_fn::<O::Output>,
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
    pub fn clone_fn(&self) -> CloneFn {
        self.clone_fn
    }
    pub fn state_drop_fn(&self) -> DropFn {
        self.state_drop_fn
    }
    pub fn output_drop_fn(&self) -> DropFn {
        self.output_drop_fn
    }

    /// Consume the init closure, producing `(state_ptr, output_template_ptr)`.
    ///
    /// # Safety
    ///
    /// `input_ptrs` must match [`input_type_ids`](Self::input_type_ids).
    pub unsafe fn init(
        self,
        input_ptrs: &[*const u8],
        timestamp: Instant,
    ) -> (*mut u8, *mut u8) {
        (self.init_fn)(input_ptrs, timestamp)
    }
}

// ---------------------------------------------------------------------------
// Monomorphised helpers
// ---------------------------------------------------------------------------

/// Monomorphised compute: materialise `Refs<'_>` and `Produced<'_>` trees
/// from the flat buffers and call `O::compute`.
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

/// Monomorphised clone of a boxed `T: Clone` into a fresh boxed `T`.
unsafe fn erased_clone_fn<T: Clone + Send + 'static>(template_ptr: *const u8) -> *mut u8 {
    let template = unsafe { &*(template_ptr as *const T) };
    Box::into_raw(Box::new(template.clone())) as *mut u8
}

/// Monomorphised drop of a boxed `T`.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
