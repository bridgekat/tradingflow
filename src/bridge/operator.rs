//! Python operator machinery for the bridge.
//!
//! [`make_py_operator`] constructs an [`ErasedOperator`] for a
//! Python-implemented operator.  [`resolve_type_id`] maps `(kind, dtype)`
//! strings to Rust [`TypeId`]s.
//!
//! The compute function ([`py_compute_fn`]) is **not** generic — it works
//! entirely through Python views stored in [`PyOperatorState`].  Dtype
//! dispatch is only needed for output allocation and view creation.

use std::any::TypeId;

use pyo3::prelude::*;

use crate::operator::{ErasedOperator, Notify};
use crate::{Array, Series};

use super::dispatch::dispatch_dtype;
use super::views::{ViewKind, create_view};
use super::{ErrorSlot, set_error};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Per-operator state for the non-generic [`py_compute_fn`].
///
/// Holds the Python callback, pre-built input/output views, mutable
/// Python state, a pre-allocated notify view, and an error slot shared
/// with the scenario.
struct PyOperatorState {
    py_operator: PyObject,
    py_inputs: PyObject,
    py_output: PyObject,
    py_notify: PyObject,
    py_state: PyObject,
    error_slot: ErrorSlot,
}

unsafe impl Send for PyOperatorState {}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Construct an [`ErasedOperator`] for a Python-implemented operator.
///
/// Allocates the output value, creates its Python view, and packages
/// everything into a type-erased operator.  Dtype dispatch is only needed
/// for the output allocation — the compute function is non-generic.
pub fn make_py_operator(
    py: Python<'_>,
    input_type_ids: Box<[TypeId]>,
    output_type_id: TypeId,
    out_dtype: &str,
    out_view_kind: ViewKind,
    output_shape: &[usize],
    py_inputs: PyObject,
    py_operator: PyObject,
    py_state: PyObject,
    error_slot: ErrorSlot,
) -> PyResult<ErasedOperator> {
    // Allocate output (generic on T) and create its Python view.
    macro_rules! alloc_output {
        ($T:ty) => {
            match out_view_kind {
                ViewKind::Array => (
                    Box::into_raw(Box::new(Array::<$T>::zeros(output_shape))) as *mut u8,
                    drop_fn::<Array<$T>> as unsafe fn(*mut u8),
                ),
                ViewKind::Series => (
                    Box::into_raw(Box::new(Series::<$T>::new(output_shape))) as *mut u8,
                    drop_fn::<Series<$T>> as unsafe fn(*mut u8),
                ),
            }
        };
    }
    let (output_ptr, output_drop_fn): (*mut u8, unsafe fn(*mut u8)) =
        dispatch_dtype!(out_dtype, alloc_output);

    let py_output = create_view(py, output_ptr, output_shape, out_dtype, out_view_kind)?;
    let py_notify = Py::new(py, super::views::NativeNotify::from_empty())?.into_any();
    let state = Box::new(PyOperatorState {
        py_operator,
        py_inputs,
        py_output,
        py_notify,
        py_state,
        error_slot,
    });

    // SAFETY: output_ptr is a valid Array<T> or Series<T>;
    // state is a valid PyOperatorState; all fn ptrs match.
    Ok(unsafe {
        ErasedOperator::new(
            TypeId::of::<PyOperatorState>(),
            input_type_ids,
            output_type_id,
            Box::new(move |_, _| (Box::into_raw(state) as *mut u8, output_ptr)),
            py_compute_fn,
            drop_fn::<PyOperatorState>,
            output_drop_fn,
        )
    })
}

// ---------------------------------------------------------------------------
// Non-generic function pointers
// ---------------------------------------------------------------------------

/// Compute function for Python operators.
///
/// Calls `operator.compute(timestamp, inputs, output, state, notify)` via
/// GIL.  Not generic — works entirely through Python views in
/// [`PyOperatorState`].
///
/// # Safety
///
/// * `state_ptr` must point to a valid `PyOperatorState`.
unsafe fn py_compute_fn(
    state_ptr: *mut u8,
    _input_ptrs: &[*const u8],
    _output_ptr: *mut u8,
    timestamp: i64,
    notify: &Notify,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut PyOperatorState) };

    if state.error_slot.lock().unwrap().is_some() {
        return false;
    }

    let result = Python::attach(|py| -> PyResult<bool> {
        // Update the pre-allocated NativeNotify with current pointers.
        {
            let mut native_notify = state
                .py_notify
                .cast_bound::<super::views::NativeNotify>(py)?
                .borrow_mut();
            unsafe { native_notify.update_from(notify) };
        }

        let result = state.py_operator.call_method1(
            py,
            "compute",
            (
                timestamp,
                &state.py_inputs,
                &state.py_output,
                &state.py_state,
                &state.py_notify,
            ),
        )?;

        let tuple = result.bind(py);
        let produced: bool = tuple.get_item(0)?.extract()?;
        let new_state = tuple.get_item(1)?;
        state.py_state = new_state.unbind();

        Ok(produced)
    });

    match result {
        Ok(produced) => produced,
        Err(e) => {
            set_error(&state.error_slot, e.to_string());
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Type-erased box drop function, monomorphised per value type.
unsafe fn drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
