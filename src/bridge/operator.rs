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

use crate::Instant;
use crate::{Array, Series};
use crate::ErasedOperator;

use super::dispatch::dispatch_dtype;
use super::views::{NativeNodeKind, create_view};
use super::{ErrorSlot, set_error};

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Per-operator state for the non-generic [`py_compute_fn`].
///
/// Holds the Python callback, pre-built input/output views, mutable
/// Python state, and an error slot shared with the scenario.
///
/// Inputs and produced are structurally symmetric on the Python side:
/// `py_inputs` is a flat tuple of views (one per input position);
/// `produced` is built fresh each compute as a flat tuple of bools with
/// the same arity.  Python operators access them identically:
/// `inputs[i]` / `produced[i]`.
struct PyOperatorState {
    py_operator: PyObject,
    py_inputs: PyObject,
    py_output: PyObject,
    py_state: PyObject,
    /// Arity of the flat input tuple; used to size `produced` on each compute.
    num_inputs: usize,
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
    out_view_kind: NativeNodeKind,
    output_shape: &[usize],
    py_inputs: PyObject,
    py_operator: PyObject,
    timestamp: Instant,
    error_slot: ErrorSlot,
) -> PyResult<ErasedOperator> {
    // Allocate output and create its Python view.
    // Unit outputs carry no data — use a 1-byte dummy allocation that is
    // never written to or read from; the Python operator receives None.
    let (output_ptr, output_drop_fn): (*mut u8, unsafe fn(*mut u8)) =
        if out_view_kind == NativeNodeKind::Unit {
            (Box::into_raw(Box::new(())) as *mut u8, drop_fn::<()>)
        } else {
            macro_rules! alloc_output {
                ($T:ty) => {
                    match out_view_kind {
                        NativeNodeKind::Array => (
                            Box::into_raw(Box::new(Array::<$T>::zeros(output_shape))) as *mut u8,
                            drop_fn::<Array<$T>> as unsafe fn(*mut u8),
                        ),
                        NativeNodeKind::Series => (
                            Box::into_raw(Box::new(Series::<$T>::new(output_shape))) as *mut u8,
                            drop_fn::<Series<$T>> as unsafe fn(*mut u8),
                        ),
                        NativeNodeKind::Unit => unreachable!(),
                    }
                };
            }
            dispatch_dtype!(out_dtype, alloc_output)
        };

    let native_output = create_view(py, output_ptr, output_shape, out_dtype, out_view_kind)?;

    // Wrap native views in Python-side wrappers (ArrayView/SeriesView).
    // Unit output: pass None directly to the Python operator.
    let views_mod = py.import("tradingflow.data.views")?;
    let py_output: PyObject = match out_view_kind {
        NativeNodeKind::Array => {
            let cls = views_mod.getattr("ArrayView")?;
            cls.call1((native_output.bind(py),))?.unbind()
        }
        NativeNodeKind::Series => {
            let cls = views_mod.getattr("SeriesView")?;
            cls.call1((native_output.bind(py),))?.unbind()
        }
        NativeNodeKind::Unit => py.None(),
    };

    // Call operator.init(inputs, timestamp) to get initial state.
    // Wire format is TAI ns (matches numpy naive `datetime64[ns]`).
    let py_state = py_operator
        .call_method1(py, "init", (&py_inputs, timestamp.as_nanos()))?;

    let num_inputs = input_type_ids.len();
    let state = Box::new(PyOperatorState {
        py_operator,
        py_inputs,
        py_output,
        py_state,
        num_inputs,
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
/// Calls `operator.compute(state, inputs, output, timestamp, produced)`
/// via GIL.  State is modified in-place by Python.  Not generic — works
/// entirely through Python objects in [`PyOperatorState`].
///
/// Builds a fresh flat `tuple[bool, ...]` for `produced` each call — same
/// arity as `inputs`, same shape, same indexing (`produced[i]` parallel
/// to `inputs[i]`).  Owned by the tuple object: safe to hold beyond
/// compute scope.
///
/// # Safety
///
/// * `state_ptr` must point to a valid `PyOperatorState`.
unsafe fn py_compute_fn(
    state_ptr: *mut u8,
    _input_ptrs: &[*const u8],
    _output_ptr: *mut u8,
    timestamp: Instant,
    produced_words: &[u64],
    produced_bit_off: usize,
    produced_num_inputs: usize,
) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut PyOperatorState) };

    if state.error_slot.lock().unwrap().is_some() {
        return false;
    }

    let n = state.num_inputs;
    debug_assert_eq!(produced_num_inputs, n);

    let result = Python::attach(|py| -> PyResult<bool> {
        // Build a flat `tuple[bool, ...]` matching the arity of `py_inputs`.
        // Each compute allocates one tuple + `n` bool refs; Python caches
        // the `True`/`False` singletons so only the tuple is new.
        let mut bits = Vec::with_capacity(n);
        let mut reader = crate::data::BitRead::from_parts(
            produced_words,
            produced_bit_off,
            produced_num_inputs,
        );
        for _ in 0..n {
            bits.push(reader.pop());
        }
        let py_produced = pyo3::types::PyTuple::new(py, &bits)?;

        let produced: bool = state
            .py_operator
            .call_method1(
                py,
                "compute",
                (
                    &state.py_state,
                    &state.py_inputs,
                    &state.py_output,
                    timestamp.as_nanos(),
                    py_produced,
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
