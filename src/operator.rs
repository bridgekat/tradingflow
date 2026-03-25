use std::any::TypeId;

use super::types::InputTypes;

/// A synchronous computation node that reads typed inputs and writes a
/// typed output.
///
/// # Lifecycle
///
/// 1. [`init`](Self::init) consumes the spec, producing runtime
///    [`State`](Self::State) and the initial [`Output`](Self::Output).
/// 2. [`compute`](Self::compute) is called on each flush with the current
///    state, input references, output reference, and timestamp.
pub trait Operator: 'static {
    /// Mutable runtime state.
    type State: Send + 'static;
    /// Input types (e.g. `(Array<f64>, Array<f64>)`).
    type Inputs: InputTypes + ?Sized;
    /// Output type.
    type Output: Send + 'static;

    /// Consume the spec and produce initial state and output.
    fn init(
        self,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        timestamp: i64,
    ) -> (Self::State, Self::Output);

    /// Compute the output from inputs and current state.
    ///
    /// Returns `true` if downstream propagation should occur.
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: i64,
    ) -> bool;
}

/// Type-erased initialization function closure.
///
/// Arguments: `(input_ptrs, timestamp)`.
/// Returns `(state_ptr, output_ptr)`, where both pointers are from
/// [`Box::into_raw`].
pub type InitFn = Box<dyn FnOnce(&[*const u8], i64) -> (*mut u8, *mut u8)>;

/// Type-erased compute function pointer.
///
/// Arguments: `(state_ptr, input_ptrs, output_ptr, timestamp)`.
/// Returns `true` if downstream propagation should occur.
pub type ComputeFn = unsafe fn(*mut u8, &[*const u8], *mut u8, i64) -> bool;

/// Type-erased representation of an operator.
///
/// # Lifecycle
///
/// 1. Created via [`from_operator`](ErasedOperator::from_operator) (safe,
///    typed) or [`new`](ErasedOperator::new) (`unsafe`, raw).
/// 2. Consumed by `Node::from_erased_operator`, which validates input
///    types, calls the deferred init, and constructs the DAG node.
/// 3. If dropped without being installed, the closure drops its captured
///    state (e.g. the operator spec) automatically.
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
    /// Construct from raw, type-erased components.
    ///
    /// # Safety
    ///
    /// * `init_fn` must correctly interpret input pointers as
    ///   `input_type_ids` types, and return valid `(state_ptr, output_ptr)`
    ///   from `Box::into_raw`, which contain types matching `state_type_id`
    ///   and `output_type_id`.
    /// * `compute_fn` must correctly interpret state pointer as `state_type_id`
    ///   type, input pointers as `input_type_ids` types and output pointer as
    ///   `output_type_id` type.
    /// * `state_drop_fn` & `output_drop_fn` must correctly drop the respective
    ///   types.
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
            input_type_ids,
            output_type_id,
            state_type_id,
            init_fn,
            compute_fn,
            state_drop_fn,
            output_drop_fn,
        }
    }

    /// Construct from a typed [`Operator`] configuration.
    pub fn from_operator<O: Operator>(op: O, arity: usize) -> Self {
        Self {
            state_type_id: TypeId::of::<O::State>(),
            input_type_ids: <O::Inputs as InputTypes>::type_ids(arity),
            output_type_id: TypeId::of::<O::Output>(),
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
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut O::State) };
    let inputs = unsafe { <O::Inputs as InputTypes>::from_ptrs(input_ptrs) };
    let output = unsafe { &mut *(output_ptr as *mut O::Output) };
    O::compute(state, inputs, output, timestamp)
}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
