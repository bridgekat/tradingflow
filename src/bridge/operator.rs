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

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::operator::ErasedOperator;
use crate::{Array, Series};

use super::dispatch::{dispatch_dtype, normalize_dtype};
use super::views::{ViewKind, create_view};
use super::{ErrorSlot, set_error};

type PyObject = Py<PyAny>;

/// Per-operator state for Python-implemented operators.
///
/// Holds the Python callback, pre-built input/output views, mutable
/// Python state, and an error slot shared with the scenario.
struct PyOperatorState {
    py_operator: PyObject,
    py_inputs: PyObject,
    py_output: PyObject,
    py_state: PyObject,
    error_slot: ErrorSlot,
}

unsafe impl Send for PyOperatorState {}

/// Resolve a Python-side `(kind, dtype)` pair to a Rust `TypeId`.
pub fn resolve_type_id(kind: &str, dtype: &str) -> PyResult<TypeId> {
    let dtype = normalize_dtype(dtype);
    macro_rules! resolve {
        ($T:ty) => {
            match kind {
                "array" => Ok(TypeId::of::<Array<$T>>()),
                "series" => Ok(TypeId::of::<Series<$T>>()),
                other => Err(PyTypeError::new_err(format!("unknown node kind: {other}"))),
            }
        };
    }
    dispatch_dtype!(dtype, resolve)
}

/// Construct an [`ErasedOperator`] for a Python-implemented operator.
///
/// Allocates the output value, creates its Python view, and packages
/// everything into a type-erased operator.  Dtype dispatch is only needed
/// for the output allocation — the compute function is non-generic.
pub fn make_py_operator(
    py: Python<'_>,
    input_type_ids: Box<[TypeId]>,
    output_type_id: TypeId,
    py_inputs: PyObject,
    py_operator: PyObject,
    py_state: PyObject,
    error_slot: ErrorSlot,
    out_dtype: &str,
    out_view_kind: ViewKind,
    output_shape: &[usize],
) -> PyResult<ErasedOperator> {
    // Only the output allocation depends on the concrete scalar type.
    macro_rules! alloc_output {
        ($T:ty) => {
            match out_view_kind {
                ViewKind::Array => (
                    Box::into_raw(Box::new(Array::<$T>::zeros(output_shape))) as *mut u8,
                    erased_drop_fn::<Array<$T>> as unsafe fn(*mut u8),
                ),
                ViewKind::Series => (
                    Box::into_raw(Box::new(Series::<$T>::new(output_shape))) as *mut u8,
                    erased_drop_fn::<Series<$T>> as unsafe fn(*mut u8),
                ),
            }
        };
    }
    let (output_ptr, output_drop_fn): (*mut u8, unsafe fn(*mut u8)) =
        dispatch_dtype!(out_dtype, alloc_output);

    let py_output = create_view(py, output_ptr, output_shape, out_dtype, out_view_kind)?;
    let state = PyOperatorState {
        py_operator,
        py_inputs,
        py_output,
        py_state,
        error_slot,
    };
    let state_ptr = Box::into_raw(Box::new(state)) as *mut u8;
    // SAFETY: output_ptr is a valid Array<T> or Series<T>;
    // state_ptr is a valid PyOperatorState; all fn ptrs match.
    Ok(unsafe {
        ErasedOperator::new(
            TypeId::of::<PyOperatorState>(),
            input_type_ids,
            output_type_id,
            Box::new(move |_, _| (state_ptr, output_ptr)),
            py_compute_fn,
            erased_drop_fn::<PyOperatorState>,
            output_drop_fn,
        )
    })
}

/// Compute function for Python operators.
///
/// Calls `operator.compute(timestamp, inputs, output, state)` via GIL.
/// Not generic — works entirely through Python views in [`PyOperatorState`].
///
/// # Safety
///
/// * `state_ptr` must point to a valid `PyOperatorState`.
unsafe fn py_compute_fn(
    state_ptr: *mut u8,
    _input_ptrs: &[*const u8],
    _output_ptr: *mut u8,
    timestamp: i64,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut PyOperatorState) };

    if state.error_slot.lock().unwrap().is_some() {
        return false;
    }

    let result = Python::attach(|py| -> PyResult<bool> {
        let result = state.py_operator.call_method1(
            py,
            "compute",
            (
                timestamp,
                &state.py_inputs,
                &state.py_output,
                &state.py_state,
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

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
