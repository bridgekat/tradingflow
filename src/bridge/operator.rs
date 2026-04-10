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

use crate::{Array, Series};
use crate::{ErasedOperator, Notify};

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
    is_clock_triggerable: bool,
    out_dtype: &str,
    out_view_kind: ViewKind,
    output_shape: &[usize],
    py_inputs: PyObject,
    py_operator: PyObject,
    timestamp: i64,
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

    let native_output = create_view(py, output_ptr, output_shape, out_dtype, out_view_kind)?;
    let native_notify = Py::new(py, super::views::NativeNotify::from_empty())?.into_any();

    // Wrap native views in Python-side wrappers (ArrayView/SeriesView/Notify).
    let views_mod = py.import("tradingflow.views")?;
    let out_wrapper_cls = match out_view_kind {
        ViewKind::Array => views_mod.getattr("ArrayView")?,
        ViewKind::Series => views_mod.getattr("SeriesView")?,
    };
    let py_output: PyObject = out_wrapper_cls
        .call1((native_output.bind(py),))?
        .unbind();
    let notify_cls = views_mod.getattr("Notify")?;
    let py_notify: PyObject = notify_cls
        .call1((native_notify.bind(py),))?
        .unbind();

    // Call operator.init(inputs, timestamp) to get initial state.
    let py_state = py_operator
        .call_method1(py, "init", (&py_inputs, timestamp))?;

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
            is_clock_triggerable,
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
/// Calls `operator.compute(state, inputs, output, timestamp, notify)` via
/// GIL.  State is modified in-place by Python.  Not generic — works
/// entirely through Python views in [`PyOperatorState`].
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
        // Update the pre-allocated NativeNotify (inside the Python Notify
        // wrapper) with current pointers.
        {
            let inner = state.py_notify.getattr(py, "_inner")?;
            let mut native_notify = inner
                .bind(py)
                .downcast::<super::views::NativeNotify>()?
                .borrow_mut();
            unsafe { native_notify.update_from(notify) };
        }

        let produced: bool = state
            .py_operator
            .call_method1(
                py,
                "compute",
                (
                    &state.py_state,
                    &state.py_inputs,
                    &state.py_output,
                    timestamp,
                    &state.py_notify,
                ),
            )?
            .extract(py)?;

        Ok(produced)
    });

    match result {
        Ok(produced) => produced,
        Err(e) => {
            set_error(&state.error_slot, e);
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
