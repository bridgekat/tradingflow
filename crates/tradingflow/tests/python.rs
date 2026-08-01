//! Tests for the input borrow contract.
//!
//! The buffer-protocol mechanics are exercised through `memoryview` and the
//! raw C-API, which need no third-party packages; the NumPy tests cover the
//! integration operators actually see and need `numpy` importable by the
//! embedded interpreter.

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

use tradingflow::data::{Array, ArrayView, Duration, Instant, Series};
use tradingflow::python::{NativeView, Scope};

// ===========================================================================
// Helpers
// ===========================================================================

/// Runs `code` with `globals`, printing any traceback before failing.
#[track_caller]
fn run(py: Python<'_>, globals: &Bound<'_, PyDict>, code: &str) {
    let code = CString::new(code).unwrap();
    py.run(code.as_c_str(), Some(globals), None)
        .unwrap_or_else(|e| {
            e.print(py);
            panic!("python snippet failed (see traceback above)");
        });
}

/// Evaluates `expr` with `globals` and extracts the result.
#[track_caller]
fn eval<T>(py: Python<'_>, globals: &Bound<'_, PyDict>, expr: &str) -> T
where
    T: for<'a, 'py> FromPyObject<'a, 'py>,
    for<'a, 'py> <T as FromPyObject<'a, 'py>>::Error: std::fmt::Debug,
{
    let expr = CString::new(expr).unwrap();
    py.eval(expr.as_c_str(), Some(globals), None)
        .unwrap_or_else(|e| {
            e.print(py);
            panic!("python expression failed (see traceback above)");
        })
        .extract()
        .expect("unexpected result type")
}

/// Acquires a buffer with raw `flags`, releasing it immediately on success.
/// Returns the raised error, if any.
fn acquire(py: Python<'_>, view: &Bound<'_, NativeView>, flags: std::ffi::c_int) -> Option<PyErr> {
    // SAFETY: `view` is a live object; the slot is released on success and
    // left untouched on failure.
    unsafe {
        let mut buffer: ffi::Py_buffer = std::mem::zeroed();
        if ffi::PyObject_GetBuffer(view.as_ptr(), &mut buffer, flags) < 0 {
            return Some(PyErr::take(py).expect("failed acquisition must set an error"));
        }
        ffi::PyBuffer_Release(&mut buffer);
        None
    }
}

/// Whether `numpy` is importable by the embedded interpreter. The NumPy tests
/// assert on this rather than skipping, so a misconfigured environment fails
/// loudly instead of silently passing.
fn numpy(py: Python<'_>) -> Bound<'_, PyModule> {
    py.import("numpy").unwrap_or_else(|e| {
        e.print(py);
        panic!(
            "numpy is not importable by the embedded interpreter; \
             point PYTHONPATH at an environment that has it"
        );
    })
}

/// A 2x3 test array with distinct elements.
fn grid() -> Array<f64, 2> {
    Array::from_parts([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into())
}

// ===========================================================================
// Buffer metadata
// ===========================================================================

#[test]
fn contiguous_array_exports_shape_and_strides() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("v", &view).unwrap();

        run(py, &globals, "m = memoryview(v)");
        assert_eq!(eval::<Vec<usize>>(py, &globals, "list(m.shape)"), [2, 3]);
        assert_eq!(eval::<Vec<isize>>(py, &globals, "list(m.strides)"), [24, 8]);
        assert_eq!(eval::<String>(py, &globals, "m.format"), "d");
        assert_eq!(eval::<usize>(py, &globals, "m.itemsize"), 8);
        assert!(eval::<bool>(py, &globals, "m.readonly"));
        assert!(eval::<bool>(py, &globals, "m.c_contiguous"));
        assert_eq!(
            eval::<Vec<Vec<f64>>>(py, &globals, "m.tolist()"),
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        );

        run(py, &globals, "del m");
        scope.close(py).unwrap();
    });
}

#[test]
fn strided_view_is_exported_without_materializing() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        // Columns 1..3 of each row: extents [2, 2], element strides [3, 1].
        let view = unsafe { scope.array(py, a.view().slice((.., 1..3))) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("v", &view).unwrap();

        run(py, &globals, "m = memoryview(v)");
        assert_eq!(eval::<Vec<usize>>(py, &globals, "list(m.shape)"), [2, 2]);
        assert_eq!(eval::<Vec<isize>>(py, &globals, "list(m.strides)"), [24, 8]);
        assert!(!eval::<bool>(py, &globals, "m.c_contiguous"));
        assert_eq!(
            eval::<Vec<Vec<f64>>>(py, &globals, "m.tolist()"),
            [[1.0, 2.0], [4.0, 5.0]],
        );

        run(py, &globals, "del m");
        scope.close(py).unwrap();
    });
}

#[test]
fn transposed_view_is_exported_as_fortran_order() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view().transpose([1, 0])) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("v", &view).unwrap();

        run(py, &globals, "m = memoryview(v)");
        assert_eq!(eval::<Vec<usize>>(py, &globals, "list(m.shape)"), [3, 2]);
        assert_eq!(eval::<Vec<isize>>(py, &globals, "list(m.strides)"), [8, 24]);
        assert!(!eval::<bool>(py, &globals, "m.c_contiguous"));
        assert!(eval::<bool>(py, &globals, "m.f_contiguous"));
        assert_eq!(
            eval::<Vec<Vec<f64>>>(py, &globals, "m.tolist()"),
            [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
        );

        run(py, &globals, "del m");
        scope.close(py).unwrap();
    });
}

#[test]
fn rank_zero_and_bool_views() {
    let signal = true;
    let scalar = 42.0_f64;
    Python::attach(|py| {
        let mut scope = Scope::new();
        let pulse = unsafe { scope.array(py, ArrayView::scalar(&signal)) }.unwrap();
        let value = unsafe { scope.array(py, ArrayView::scalar(&scalar)) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("pulse", &pulse).unwrap();
        globals.set_item("value", &value).unwrap();

        run(py, &globals, "p, x = memoryview(pulse), memoryview(value)");
        assert_eq!(eval::<usize>(py, &globals, "p.ndim"), 0);
        assert_eq!(eval::<String>(py, &globals, "p.format"), "?");
        assert!(eval::<bool>(py, &globals, "p.tolist()"));
        assert_eq!(eval::<usize>(py, &globals, "x.ndim"), 0);
        assert_eq!(eval::<f64>(py, &globals, "x.tolist()"), 42.0);

        run(py, &globals, "del p, x");
        scope.close(py).unwrap();
    });
}

#[test]
fn series_binds_values_and_instants() {
    let instants: Vec<Instant> = (0..4)
        .map(|i| Instant::from_offset(Duration::from_days(i)))
        .collect();
    let s = Series::from_parts([2], instants, (0..8).map(|i| i as f64).collect(), 0);

    Python::attach(|py| {
        let mut scope = Scope::new();
        let (ts, values) = unsafe { scope.series(py, s.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("ts", &ts).unwrap();
        globals.set_item("values", &values).unwrap();

        run(py, &globals, "mv, mt = memoryview(values), memoryview(ts)");
        // The time axis is prepended, so a rank-1 element series reads rank 2.
        assert_eq!(eval::<Vec<usize>>(py, &globals, "list(mv.shape)"), [4, 2]);
        assert_eq!(
            eval::<Vec<isize>>(py, &globals, "list(mv.strides)"),
            [16, 8]
        );
        assert_eq!(
            eval::<Vec<Vec<f64>>>(py, &globals, "mv.tolist()"),
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        );
        // Timestamps travel as int64 nanoseconds.
        assert_eq!(eval::<String>(py, &globals, "mt.format"), "q");
        assert_eq!(
            eval::<Vec<i64>>(py, &globals, "mt.tolist()"),
            (0..4).map(|i| i * 86_400_000_000_000).collect::<Vec<_>>(),
        );

        run(py, &globals, "del mv, mt");
        scope.close(py).unwrap();
    });
}

// ===========================================================================
// Request negotiation
// ===========================================================================

#[test]
fn writable_requests_are_refused() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();

        let err = acquire(py, &view, ffi::PyBUF_WRITABLE).expect("writable must be refused");
        assert!(err.is_instance_of::<PyBufferError>(py));
        assert!(err.to_string().contains("read-only"));
        assert_eq!(view.get().exports(), 0);

        scope.close(py).unwrap();
    });
}

#[test]
fn simple_requests_require_contiguity() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let contiguous = unsafe { scope.array(py, a.view()) }.unwrap();
        let strided = unsafe { scope.array(py, a.view().slice((.., 1..3))) }.unwrap();

        // A flat request is fine for a contiguous view...
        assert!(acquire(py, &contiguous, ffi::PyBUF_SIMPLE).is_none());
        // ...but a strided one cannot be read flat.
        let err = acquire(py, &strided, ffi::PyBUF_SIMPLE).expect("strided must be refused");
        assert!(err.is_instance_of::<PyBufferError>(py));
        assert!(err.to_string().contains("PyBUF_STRIDES"));
        // Likewise for an explicit C-contiguity demand.
        let err = acquire(py, &strided, ffi::PyBUF_C_CONTIGUOUS).expect("strided must be refused");
        assert!(err.is_instance_of::<PyBufferError>(py));

        scope.close(py).unwrap();
    });
}

#[test]
fn acquisition_balances_the_export_counter() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();

        assert_eq!(view.get().exports(), 0);
        assert!(acquire(py, &view, ffi::PyBUF_FULL_RO).is_none());
        assert_eq!(view.get().exports(), 0);

        scope.close(py).unwrap();
    });
}

// ===========================================================================
// The contract
// ===========================================================================

#[test]
fn transient_consumers_do_not_escape() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("v", &view).unwrap();

        // Read it, copy what we keep, let the consumer die with the frame.
        run(
            py,
            &globals,
            "def compute(v):\n    return sum(memoryview(v).tolist()[0])\nkept = compute(v)",
        );
        assert_eq!(eval::<f64>(py, &globals, "kept"), 3.0);
        assert_eq!(view.get().exports(), 0);

        scope.close(py).expect("no pointer escaped");
    });
}

#[test]
fn retained_consumer_is_reported_as_an_escape() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("v", &view).unwrap();

        // The classic accident: memoize a slice of this tick's input.
        run(py, &globals, "state = {}\nstate['prev'] = memoryview(v)");
        assert_eq!(view.get().exports(), 1);

        let escape = scope.close(py).expect_err("the export must be reported");
        assert_eq!(escape.len(), 1);
        let message = escape.to_string();
        assert!(message.contains("view #0"));
        assert!(message.contains("float64[2, 3]"));
        assert!(message.contains("1 outstanding buffer export"));
    });
}

#[test]
fn retained_view_object_is_harmless_but_expires() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("v", &view).unwrap();

        // Keeping the view object itself captures no pointer, so it is not an
        // escape...
        run(py, &globals, "state = {'view': v}");
        scope.close(py).expect("no pointer escaped");

        // ...but the borrow window has closed, so using it later fails loudly
        // rather than reading memory the graph has moved on from.
        assert!(!view.get().valid());
        let err = acquire(py, &view, ffi::PyBUF_FULL_RO).expect("expired view must refuse");
        assert!(err.is_instance_of::<PyBufferError>(py));
        assert!(err.to_string().contains("expired"));
    });
}

#[test]
fn exports_trapped_in_a_cycle_are_collected() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("v", &view).unwrap();

        // A self-referential local: unreachable when the frame exits, but the
        // cycle collector has to run before its export is released.
        run(
            py,
            &globals,
            "def compute(v):\n\
             \x20   cycle = {'buffer': memoryview(v)}\n\
             \x20   cycle['self'] = cycle\n\
             compute(v)",
        );

        // `close` collects once and re-checks rather than reporting a false
        // escape.
        scope.close(py).expect("a dead cycle is not an escape");
    });
}

#[test]
fn every_view_in_a_scope_is_checked() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let _first = unsafe { scope.array(py, a.view()) }.unwrap();
        let second = unsafe { scope.array(py, ArrayView::scalar(&1.0_f64)) }.unwrap();
        let third = unsafe { scope.array(py, a.view().slice((.., 1..3))) }.unwrap();
        assert_eq!(scope.len(), 3);

        let globals = PyDict::new(py);
        globals.set_item("second", &second).unwrap();
        globals.set_item("third", &third).unwrap();
        run(
            py,
            &globals,
            "state = [memoryview(second), memoryview(third)]",
        );

        let escape = scope.close(py).expect_err("both exports must be reported");
        assert_eq!(escape.len(), 2);
        let message = escape.to_string();
        assert!(message.contains("view #1"));
        assert!(message.contains("view #2"));
        assert!(!message.contains("view #0"));
    });
}

// ===========================================================================
// NumPy integration
// ===========================================================================

#[test]
fn numpy_reads_the_view_zero_copy() {
    let a = grid();
    let address = a.data().as_ptr() as usize;
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("np", numpy(py)).unwrap();
        globals.set_item("v", &view).unwrap();

        run(py, &globals, "arr = np.asarray(v)");
        assert_eq!(eval::<Vec<usize>>(py, &globals, "list(arr.shape)"), [2, 3]);
        assert_eq!(eval::<String>(py, &globals, "str(arr.dtype)"), "float64");
        assert_eq!(eval::<f64>(py, &globals, "float(arr.sum())"), 15.0);

        // Zero-copy: the array points straight at the Rust buffer...
        assert_eq!(
            eval::<usize>(py, &globals, "arr.__array_interface__['data'][0]"),
            address,
        );
        // ...read-only, so no write provenance ever reaches Python...
        assert!(!eval::<bool>(py, &globals, "arr.flags.writeable"));
        run(
            py,
            &globals,
            "try:\n\
             \x20   arr[0, 0] = 1.0\n\
             \x20   raised = False\n\
             except ValueError:\n\
             \x20   raised = True",
        );
        assert!(eval::<bool>(py, &globals, "raised"));
        // ...and the array holds the export that proves it is still borrowing.
        assert_eq!(view.get().exports(), 1);

        run(py, &globals, "del arr");
        assert_eq!(view.get().exports(), 0);
        scope.close(py).unwrap();
    });
}

#[test]
fn numpy_reads_a_strided_view_correctly() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view().slice((.., 1..3))) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("np", numpy(py)).unwrap();
        globals.set_item("v", &view).unwrap();

        run(py, &globals, "arr = np.asarray(v)");
        assert_eq!(
            eval::<Vec<Vec<f64>>>(py, &globals, "arr.tolist()"),
            [[1.0, 2.0], [4.0, 5.0]],
        );
        // Still a view of the graph's memory, not a repacked copy.
        assert!(!eval::<bool>(py, &globals, "arr.flags.owndata"));

        run(py, &globals, "del arr");
        scope.close(py).unwrap();
    });
}

#[test]
fn numpy_recovers_datetime64_from_instants() {
    let instants: Vec<Instant> = (0..3)
        .map(|i| Instant::from_offset(Duration::from_days(i)))
        .collect();
    let s = Series::from_parts([1], instants, vec![10.0, 20.0, 30.0], 0);

    Python::attach(|py| {
        let mut scope = Scope::new();
        let (ts, values) = unsafe { scope.series(py, s.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("np", numpy(py)).unwrap();
        globals.set_item("ts", &ts).unwrap();
        globals.set_item("values", &values).unwrap();

        // The int64 -> datetime64[ns] reinterpretation is itself zero-copy.
        run(
            py,
            &globals,
            "stamps = np.asarray(ts).view('datetime64[ns]')\nrows = np.asarray(values)",
        );
        assert_eq!(
            eval::<String>(py, &globals, "str(stamps.dtype)"),
            "datetime64[ns]",
        );
        assert_eq!(
            eval::<String>(py, &globals, "str(stamps[1])"),
            "1970-01-02T00:00:00.000000000",
        );
        assert_eq!(eval::<Vec<usize>>(py, &globals, "list(rows.shape)"), [3, 1]);
        assert_eq!(eval::<f64>(py, &globals, "float(rows.sum())"), 60.0);

        run(py, &globals, "del stamps, rows");
        scope.close(py).unwrap();
    });
}

#[test]
fn numpy_copies_release_the_borrow() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("np", numpy(py)).unwrap();
        globals.set_item("v", &view).unwrap();

        // The sanctioned way to memoize an input: copy it.
        run(py, &globals, "state = {'prev': np.asarray(v).copy()}");
        assert_eq!(view.get().exports(), 0);
        assert!(eval::<bool>(py, &globals, "state['prev'].flags.owndata"));

        scope.close(py).expect("a copy is not a borrow");

        // The copy stays readable after the borrow window closes.
        assert_eq!(
            eval::<f64>(py, &globals, "float(state['prev'].sum())"),
            15.0
        );
    });
}

#[test]
fn a_returned_input_passthrough_is_legal_once_dropped() {
    let a = grid();
    Python::attach(|py| {
        let mut scope = Scope::new();
        let view = unsafe { scope.array(py, a.view()) }.unwrap();
        let globals = PyDict::new(py);
        globals.set_item("np", numpy(py)).unwrap();
        globals.set_item("v", &view).unwrap();

        // An operator returning one of its inputs unchanged.
        run(
            py,
            &globals,
            "def compute(v):\n    return np.asarray(v)\nout = compute(v)",
        );
        assert_eq!(view.get().exports(), 1);

        // The host copies the output into its own buffer, then drops it — which
        // is what makes the pass-through legal under the copy-out rule.
        let copied: f64 = eval(py, &globals, "float(np.asarray(out).sum())");
        assert_eq!(copied, 15.0);
        run(py, &globals, "del out");

        scope.close(py).expect("the pass-through was released");
    });
}
