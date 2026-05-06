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

use crate::ErasedOperator;
use crate::Instant;
use crate::{Array, Series};

use super::dispatch::dispatch_dtype;
use super::views::{NativeNodeKind, create_view};
use super::{ErrorSlot, set_error};

type PyObject = Py<PyAny>;

/// Per-input Python metadata captured by an operator's [`InitFn`] closure
/// so it can rebuild fresh input views on every session start.
#[derive(Clone)]
pub struct PyInputInfo {
    pub kind: NativeNodeKind,
    pub dtype: String,
    pub shape: Vec<usize>,
}

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
/// Packages the operator spec into a type-erased operator whose [`InitFn`]
/// is `Fn`: every invocation allocates a fresh output value, builds fresh
/// Python views for both inputs and output (using the input pointers
/// supplied by the [`Session`](crate::scenario::Session) plus the
/// captured `input_metadata`), and calls the Python operator's
/// `init(...)` to produce fresh state.  The descriptor itself is therefore
/// reusable across multiple scenario sessions — the Python operator is
/// treated as a pure configuration carrier and `init()` is expected to
/// return a fresh state object on each call.
///
/// `input_metadata` carries the per-input `(kind, dtype, shape)` tuple
/// the closure needs to wrap each raw pointer in a typed
/// `NativeArrayView` / `NativeSeriesView` (and then in the Python-side
/// `ArrayView` / `SeriesView`) on every session start.
///
/// Dtype dispatch is only needed for the output allocation — the compute
/// function is non-generic.
pub fn make_py_operator(
    input_type_ids: Box<[TypeId]>,
    input_metadata: Vec<PyInputInfo>,
    output_type_id: TypeId,
    out_dtype: &str,
    out_view_kind: NativeNodeKind,
    output_shape: &[usize],
    py_operator: PyObject,
    error_slot: ErrorSlot,
) -> PyResult<ErasedOperator> {
    assert_eq!(
        input_type_ids.len(),
        input_metadata.len(),
        "input_type_ids / input_metadata arity mismatch",
    );

    // Resolve the per-init allocator for the output value.  The dtype
    // dispatch happens here once; the resulting function pointer is then
    // called inside the (Fn) init closure on every session start.
    //
    // Unit outputs carry no data — use a 1-byte dummy allocation that is
    // never written to or read from; the Python operator receives None.
    let (alloc_output_fn, output_drop_fn): (
        fn(&[usize]) -> *mut u8,
        unsafe fn(*mut u8),
    ) = if out_view_kind == NativeNodeKind::Unit {
        (alloc_unit, drop_fn::<()>)
    } else {
        macro_rules! resolve_alloc {
            ($T:ty) => {
                match out_view_kind {
                    NativeNodeKind::Array => (
                        alloc_array::<$T> as fn(&[usize]) -> *mut u8,
                        drop_fn::<Array<$T>> as unsafe fn(*mut u8),
                    ),
                    NativeNodeKind::Series => (
                        alloc_series::<$T> as fn(&[usize]) -> *mut u8,
                        drop_fn::<Series<$T>> as unsafe fn(*mut u8),
                    ),
                    NativeNodeKind::Unit => unreachable!(),
                }
            };
        }
        dispatch_dtype!(out_dtype, resolve_alloc)
    };

    // Owned, clone-able captures for the closure.
    let out_dtype_owned = out_dtype.to_string();
    let output_shape_owned: Vec<usize> = output_shape.to_vec();
    let num_inputs = input_type_ids.len();

    let init_fn: Box<dyn Fn(&[*const u8], Instant) -> (*mut u8, *mut u8)> =
        Box::new(move |input_ptrs: &[*const u8], timestamp: Instant| {
            assert_eq!(
                input_ptrs.len(),
                input_metadata.len(),
                "py operator init: input arity mismatch",
            );
            let output_ptr = alloc_output_fn(&output_shape_owned);

            // Build fresh input + output Python views, wrap them in the
            // user-facing view classes, and call the Python operator's
            // `init(...)` — all under a single GIL acquire.  `create_view`
            // only fails on dtype/view-kind mismatches that the caller
            // has already validated.
            let (py_output_wrapped, py_state, py_inputs, py_operator_clone) =
                Python::attach(|py| -> PyResult<_> {
                    let views_mod = py.import("tradingflow.data.views")?;
                    let array_view_cls = views_mod.getattr("ArrayView")?;
                    let series_view_cls = views_mod.getattr("SeriesView")?;

                    // Fresh input views from this session's buffer pointers.
                    let input_views: Vec<PyObject> = input_metadata
                        .iter()
                        .zip(input_ptrs.iter())
                        .map(|(info, &ptr)| {
                            if info.kind == NativeNodeKind::Unit {
                                Ok(py.None())
                            } else {
                                let native = create_view(
                                    py,
                                    ptr as *mut u8,
                                    &info.shape,
                                    &info.dtype,
                                    info.kind,
                                )?;
                                let wrapped: PyObject = match info.kind {
                                    NativeNodeKind::Array => {
                                        array_view_cls.call1((native.bind(py),))?.unbind()
                                    }
                                    NativeNodeKind::Series => {
                                        series_view_cls.call1((native.bind(py),))?.unbind()
                                    }
                                    NativeNodeKind::Unit => py.None(),
                                };
                                Ok(wrapped)
                            }
                        })
                        .collect::<PyResult<_>>()?;
                    let py_inputs: PyObject = pyo3::types::PyTuple::new(py, &input_views)?
                        .into_any()
                        .unbind();

                    // Fresh output view.
                    let py_output_wrapped: PyObject = match out_view_kind {
                        NativeNodeKind::Unit => py.None(),
                        _ => {
                            let native = create_view(
                                py,
                                output_ptr,
                                &output_shape_owned,
                                &out_dtype_owned,
                                out_view_kind,
                            )?;
                            match out_view_kind {
                                NativeNodeKind::Array => {
                                    array_view_cls.call1((native.bind(py),))?.unbind()
                                }
                                NativeNodeKind::Series => {
                                    series_view_cls.call1((native.bind(py),))?.unbind()
                                }
                                NativeNodeKind::Unit => unreachable!(),
                            }
                        }
                    };

                    let py_state = py_operator.call_method1(
                        py,
                        "init",
                        (&py_inputs, timestamp.as_nanos()),
                    )?;
                    Ok((
                        py_output_wrapped,
                        py_state,
                        py_inputs,
                        py_operator.clone_ref(py),
                    ))
                })
                .expect("make_py_operator: init failed during session start");

            let state = Box::new(PyOperatorState {
                py_operator: py_operator_clone,
                py_inputs,
                py_output: py_output_wrapped,
                py_state,
                num_inputs,
                error_slot: error_slot.clone(),
            });
            (Box::into_raw(state) as *mut u8, output_ptr)
        });

    // SAFETY: init_fn returns valid PyOperatorState and output pointers;
    // all fn ptrs match.
    Ok(unsafe {
        ErasedOperator::new(
            TypeId::of::<PyOperatorState>(),
            input_type_ids,
            output_type_id,
            init_fn,
            py_compute_fn,
            drop_fn::<PyOperatorState>,
            output_drop_fn,
        )
    })
}

// ---------------------------------------------------------------------------
// Per-init output allocators
// ---------------------------------------------------------------------------

fn alloc_unit(_shape: &[usize]) -> *mut u8 {
    Box::into_raw(Box::new(())) as *mut u8
}

fn alloc_array<T: crate::Scalar>(shape: &[usize]) -> *mut u8 {
    Box::into_raw(Box::new(Array::<T>::zeros(shape))) as *mut u8
}

fn alloc_series<T: crate::Scalar>(shape: &[usize]) -> *mut u8 {
    Box::into_raw(Box::new(Series::<T>::new(shape))) as *mut u8
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
