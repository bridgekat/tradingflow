//! Tests for the copying FFI boundary.
//!
//! Both directions are exercised through NumPy, which the embedded
//! interpreter must be able to import — the helpers fail loudly on a
//! misconfigured environment rather than skipping.

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

use tradingflow::data::{Array, ArrayView, Duration, Instant, Series, SeriesView};
use tradingflow::graph::{Builder, Operator, Pool};
use tradingflow::operators::series::record_all;
use tradingflow::ports::{ArrayPort, ArrayPorts, SignalPort};
use tradingflow::python::{PyInterface, PyOperator, from_numpy_into, py_operator_source, to_numpy};
use tradingflow::sources::sync::array_series;
use tradingflow::time::UnixTime;

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

/// Evaluates `expr` with `globals`, returning the raw object.
#[track_caller]
fn eval_any<'py>(py: Python<'py>, globals: &Bound<'py, PyDict>, expr: &str) -> Bound<'py, PyAny> {
    let expr = CString::new(expr).unwrap();
    py.eval(expr.as_c_str(), Some(globals), None)
        .unwrap_or_else(|e| {
            e.print(py);
            panic!("python expression failed (see traceback above)");
        })
}

/// Evaluates `expr` with `globals` and extracts the result.
#[track_caller]
fn eval<T>(py: Python<'_>, globals: &Bound<'_, PyDict>, expr: &str) -> T
where
    T: for<'a, 'py> FromPyObject<'a, 'py>,
    for<'a, 'py> <T as FromPyObject<'a, 'py>>::Error: std::fmt::Debug,
{
    eval_any(py, globals, expr)
        .extract()
        .expect("unexpected result type")
}

/// Globals with `numpy` imported, failing loudly if it is unavailable.
fn globals(py: Python<'_>) -> Bound<'_, PyDict> {
    let np = py.import("numpy").unwrap_or_else(|e| {
        e.print(py);
        panic!(
            "numpy is not importable by the embedded interpreter; \
             point PYTHONPATH at an environment that has it"
        );
    });
    let globals = PyDict::new(py);
    globals.set_item("np", np).unwrap();
    globals
}

/// A 2x3 test array with distinct elements.
fn grid() -> Array<f64, 2> {
    Array::from_parts([2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into())
}

/// Nanosecond instants.
fn tss(xs: &[i64]) -> Vec<Instant> {
    xs.iter()
        .copied()
        .map(|ns| Instant::from_offset(Duration::from_nanos(ns)))
        .collect()
}

// ===========================================================================
// Rust to Python
// ===========================================================================

#[test]
fn arrays_copy_out_with_shape_and_dtype() {
    let a = grid();
    Python::attach(|py| {
        let globals = globals(py);
        globals.set_item("arr", to_numpy(py, a.view())).unwrap();

        assert_eq!(eval::<Vec<usize>>(py, &globals, "list(arr.shape)"), [2, 3]);
        assert_eq!(eval::<String>(py, &globals, "str(arr.dtype)"), "float64");
        assert_eq!(
            eval::<Vec<Vec<f64>>>(py, &globals, "arr.tolist()"),
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        );
    });
}

#[test]
fn strided_and_transposed_views_materialize_row_major() {
    let a = grid();
    Python::attach(|py| {
        let globals = globals(py);
        // Columns 1..3 of each row, and the transpose: both are strided views
        // Rust-side, but arrive as plain owned row-major arrays.
        globals
            .set_item("sliced", to_numpy(py, a.view().slice((.., 1..3))))
            .unwrap();
        globals
            .set_item("t", to_numpy(py, a.view().transpose([1, 0])))
            .unwrap();

        assert_eq!(
            eval::<Vec<Vec<f64>>>(py, &globals, "sliced.tolist()"),
            [[1.0, 2.0], [4.0, 5.0]],
        );
        assert!(eval::<bool>(py, &globals, "sliced.flags.c_contiguous"));
        assert_eq!(
            eval::<Vec<Vec<f64>>>(py, &globals, "t.tolist()"),
            [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
        );
        assert!(eval::<bool>(py, &globals, "t.flags.c_contiguous"));
    });
}

#[test]
fn rank_zero_and_bool_arrays() {
    Python::attach(|py| {
        let globals = globals(py);
        globals
            .set_item("pulse", to_numpy(py, ArrayView::scalar(&true)))
            .unwrap();
        globals
            .set_item("x", to_numpy(py, ArrayView::scalar(&42.0_f64)))
            .unwrap();

        assert_eq!(eval::<usize>(py, &globals, "pulse.ndim"), 0);
        assert_eq!(eval::<String>(py, &globals, "str(pulse.dtype)"), "bool");
        assert!(eval::<bool>(py, &globals, "bool(pulse)"));
        assert_eq!(eval::<usize>(py, &globals, "x.ndim"), 0);
        assert_eq!(eval::<f64>(py, &globals, "float(x)"), 42.0);
    });
}

#[test]
fn python_owns_the_copy() {
    let a = grid();
    Python::attach(|py| {
        let globals = globals(py);
        globals.set_item("arr", to_numpy(py, a.view())).unwrap();

        // The array's memory belongs to the Python object graph — the copied
        // buffer rides along as the array's `base` (so OWNDATA is false, as
        // for any base-backed array), and it is writable. Storing and mutating
        // it is legal, which is the whole point of the copying model.
        assert!(eval::<bool>(py, &globals, "arr.base is not None"));
        assert!(eval::<bool>(py, &globals, "arr.flags.writeable"));
        run(py, &globals, "state = {'prev': arr}\narr[0, 0] = 99.0");

        // The mutation stayed on the Python side.
        assert_eq!(a.view()[[0, 0]], 0.0);
    });
}

#[test]
fn the_copy_outlives_the_rust_buffer() {
    let a = grid();
    Python::attach(|py| {
        let globals = globals(py);
        globals.set_item("arr", to_numpy(py, a.view())).unwrap();

        // Free the Rust-side buffer while the Python array is alive; the copy
        // owns its memory, so nothing dangles by construction.
        drop(a);
        assert_eq!(eval::<f64>(py, &globals, "float(arr.sum())"), 15.0);
    });
}

// ===========================================================================
// Python to Rust
// ===========================================================================

#[test]
fn arrays_copy_in_from_ndarray() {
    let mut out = Array::<f64, 2>::zeros([2, 3]);
    Python::attach(|py| {
        let globals = globals(py);
        let value = eval_any(py, &globals, "np.arange(6, dtype='float64') * 10");
        from_numpy_into(&value, &mut out).unwrap();
    });
    assert_eq!(
        out,
        Array::from_parts([2, 3], vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0].into()),
    );
}

#[test]
fn lists_scalars_and_other_dtypes_are_normalized() {
    Python::attach(|py| {
        let globals = globals(py);

        // A plain list.
        let mut out = Array::<f64, 1>::zeros([3]);
        from_numpy_into(&eval_any(py, &globals, "[1, 2, 3]"), &mut out).unwrap();
        assert_eq!(out, Array::from_parts([3], vec![1.0, 2.0, 3.0].into()));

        // A Python scalar into a rank-0 output.
        let mut out = Array::<f64, 0>::zeros([]);
        from_numpy_into(&eval_any(py, &globals, "2.5"), &mut out).unwrap();
        assert_eq!(out, Array::from_parts([], vec![2.5].into()));

        // An integer ndarray is cast to the output element type.
        let mut out = Array::<f64, 1>::zeros([3]);
        from_numpy_into(&eval_any(py, &globals, "np.array([7, 8, 9])"), &mut out).unwrap();
        assert_eq!(out, Array::from_parts([3], vec![7.0, 8.0, 9.0].into()));

        // A bool output accepts a bool array.
        let mut out = Array::<bool, 1>::zeros([2]);
        from_numpy_into(&eval_any(py, &globals, "np.array([True, False])"), &mut out).unwrap();
        assert_eq!(out, Array::from_parts([2], vec![true, false].into()));
    });
}

#[test]
fn non_contiguous_values_are_repacked() {
    let mut out = Array::<f64, 2>::zeros([2, 2]);
    Python::attach(|py| {
        let globals = globals(py);
        // A Fortran-order array goes through ascontiguousarray.
        let value = eval_any(
            py,
            &globals,
            "np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))",
        );
        from_numpy_into(&value, &mut out).unwrap();
    });
    assert_eq!(
        out,
        Array::from_parts([2, 2], vec![1.0, 2.0, 3.0, 4.0].into()),
    );
}

#[test]
fn element_counts_negotiate_but_must_match() {
    Python::attach(|py| {
        let globals = globals(py);

        // Same count, different shape: accepted (the boundary negotiates by
        // element count, exactly as the output extents do).
        let mut out = Array::<f64, 2>::zeros([3, 1]);
        from_numpy_into(&eval_any(py, &globals, "[1.0, 2.0, 3.0]"), &mut out).unwrap();
        assert_eq!(out.view().data(), [1.0, 2.0, 3.0]);

        // Wrong count: rejected, and the output is untouched.
        let mut out = Array::<f64, 1>::zeros([3]);
        let err = from_numpy_into(&eval_any(py, &globals, "[1.0, 2.0]"), &mut out).unwrap_err();
        assert!(err.to_string().contains("expected 3 elements, got 2"));
        assert_eq!(out, Array::zeros([3]));

        // Something NumPy cannot coerce at all: the normalization error
        // surfaces as the Python exception it is.
        let mut out = Array::<f64, 1>::zeros([1]);
        assert!(from_numpy_into(&eval_any(py, &globals, "object()"), &mut out).is_err());
    });
}

#[test]
fn the_copy_severs_the_python_reference() {
    let mut out = Array::<f64, 1>::zeros([3]);
    Python::attach(|py| {
        let globals = globals(py);
        run(py, &globals, "src = np.array([1.0, 2.0, 3.0])");
        let value = eval_any(py, &globals, "src");
        from_numpy_into(&value, &mut out).unwrap();

        // Mutating the source afterwards must not reach the Rust copy.
        run(py, &globals, "src[:] = -1.0");
    });
    assert_eq!(out, Array::from_parts([3], vec![1.0, 2.0, 3.0].into()));
}

#[test]
fn a_round_trip_is_the_identity() {
    let a = grid();
    let mut out = Array::<f64, 2>::zeros([2, 3]);
    Python::attach(|py| {
        let globals = globals(py);
        globals.set_item("arr", to_numpy(py, a.view())).unwrap();
        // An operator returning one of its inputs unchanged: with owned
        // copies this is unremarkable rather than a special-cased hazard.
        let value = eval_any(py, &globals, "arr");
        from_numpy_into(&value, &mut out).unwrap();
    });
    assert_eq!(a, out);
}

// ===========================================================================
// PyInterface
// ===========================================================================

#[test]
fn interface_leaf_allocates_from_first_return() {
    type Leaf = ArrayPort<f64, 2>;
    Python::attach(|py| {
        let globals = globals(py);
        let mut buffers = <Leaf as PyInterface>::Buffers::default();

        // The first return fixes the rank-2 extents...
        let value = eval_any(py, &globals, "np.ones((2, 3))");
        let view = <Leaf as PyInterface>::from_python(&value, &mut buffers).unwrap();
        assert_eq!(view.extents(), [2, 3]);

        // ...a later same-count return reflows into them...
        let value = eval_any(py, &globals, "np.arange(6.0)");
        let view = <Leaf as PyInterface>::from_python(&value, &mut buffers).unwrap();
        assert_eq!(view.extents(), [2, 3]);
        assert_eq!(view[[1, 0]], 3.0);

        // ...and rank-2 extents cannot be inferred from a 1-d first return.
        let mut fresh = <Leaf as PyInterface>::Buffers::default();
        let value = eval_any(py, &globals, "np.arange(6.0)");
        assert!(<Leaf as PyInterface>::from_python(&value, &mut fresh).is_err());
    });
}

#[test]
fn interface_tuple_takes_a_sequence_of_matching_arity() {
    type Pair = (SignalPort<0>, ArrayPort<f64, 1>);
    Python::attach(|py| {
        let globals = globals(py);
        let mut buffers = <Pair as PyInterface>::Buffers::default();

        let value = eval_any(py, &globals, "(True, [1.0, 2.0])");
        let (sig, values) = <Pair as PyInterface>::from_python(&value, &mut buffers).unwrap();
        assert!(*sig);
        assert_eq!(values.data(), [1.0, 2.0]);

        let value = eval_any(py, &globals, "(False,)");
        let err = <Pair as PyInterface>::from_python(&value, &mut buffers).unwrap_err();
        assert!(err.to_string().contains("expected a sequence of 2"));
    });
}

#[test]
fn interface_group_outputs_fix_their_length() {
    type Group = ArrayPorts<f64, 1>;
    Python::attach(|py| {
        let globals = globals(py);
        let mut buffers = <Group as PyInterface>::Buffers::default();

        // The first return fixes the group's length (and each element's
        // extents)...
        let value = eval_any(py, &globals, "[np.array([1.0, 2.0]), np.array([3.0])]");
        let views = <Group as PyInterface>::from_python(&value, &mut buffers).unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].data(), [1.0, 2.0]);
        assert_eq!(views[1].data(), [3.0]);

        // ...later returns of the same length update in place...
        let value = eval_any(py, &globals, "[np.array([9.0, 8.0]), np.array([7.0])]");
        let views = <Group as PyInterface>::from_python(&value, &mut buffers).unwrap();
        assert_eq!(views[0].data(), [9.0, 8.0]);
        assert_eq!(views[1].data(), [7.0]);

        // ...and a changed length is refused.
        let value = eval_any(py, &globals, "[np.array([1.0])]");
        let err = <Group as PyInterface>::from_python(&value, &mut buffers).unwrap_err();
        assert!(err.to_string().contains("expected 2 values"));
    });
}

#[test]
fn interface_group_inputs_flatten_in_tree_order() {
    type Inputs = (SignalPort<0>, ArrayPorts<f64, 1>);
    let a = Array::from_parts([2], vec![1.0, 2.0].into());
    let b = Array::from_parts([1], vec![3.0].into());
    Python::attach(|py| {
        let mut args = Vec::new();
        let group = [a.view(), b.view()];
        <Inputs as PyInterface>::to_python(py, (ArrayView::scalar(&true), &group), &mut args)
            .unwrap();

        // One arg per leaf: the signal, then each group element.
        assert_eq!(args.len(), 3);
        let globals = globals(py);
        globals.set_item("args", args).unwrap();
        assert!(eval::<bool>(py, &globals, "bool(args[0])"));
        assert_eq!(
            eval::<Vec<f64>>(py, &globals, "args[1].tolist()"),
            [1.0, 2.0],
        );
        assert_eq!(eval::<Vec<f64>>(py, &globals, "args[2].tolist()"), [3.0]);
    });
}

// ===========================================================================
// PyOperator
// ===========================================================================

/// A stateful operator mirroring the `Operator` trait: `init` builds the state,
/// `compute` accumulates into it, `reset` returns the retained batch.
const ACCUMULATE: &str = r#"
class Accumulate:
    def __init__(self, scale):
        self.scale = scale

    def init(self, inputs):
        (sig, x) = inputs
        return {"sum": np.zeros_like(x)}

    def reset(self, inputs, state):
        return (False, state["sum"])

    def compute(self, inputs, state, instant):
        (sig, x) = inputs
        state["sum"] = state["sum"] + x * self.scale
        return (bool(sig), state["sum"])

def build(scale=1.0):
    return Accumulate(scale)
"#;

#[test]
fn operator_mirrors_the_rust_contract() {
    type In = (SignalPort<0>, ArrayPort<f64, 1>);
    type Out = (SignalPort<0>, ArrayPort<f64, 1>);
    type Op = PyOperator<In, Out>;

    // Keyword arguments for `build(**kwargs)` are a plain Python dict, built
    // with PyO3 — no bespoke params type, so any Python value works.
    let params = Python::attach(|py| {
        use pyo3::types::IntoPyDict;
        [("scale", 2.0)].into_py_dict(py).unwrap().unbind()
    });
    let op: Op = py_operator_source(ACCUMULATE, Some(params));
    let x = Array::from_parts([3], vec![1.0, 2.0, 3.0].into());
    let quiet = (ArrayView::scalar(&false), x.view());
    let firing = (ArrayView::scalar(&true), x.view());

    // Build-time: init creates the Python state, the first reset sizes the
    // output buffers from its return.
    let mut state = Operator::init(op, quiet);
    let (sig, values) = <Op as Operator>::reset(quiet, &mut state);
    assert!(!*sig);
    assert_eq!(values.data(), [0.0, 0.0, 0.0]);

    // Two computes accumulate through the carried Python state.
    let t = Instant::from_offset(Duration::from_nanos(1));
    let (sig, values) = <Op as Operator>::compute(firing, &mut state, &t);
    assert!(*sig);
    assert_eq!(values.data(), [2.0, 4.0, 6.0]);
    let (sig, values) = <Op as Operator>::compute(firing, &mut state, &t);
    assert!(*sig);
    assert_eq!(values.data(), [4.0, 8.0, 12.0]);

    // Reset retains the accumulated batch and quiesces the signal.
    let (sig, values) = <Op as Operator>::reset(quiet, &mut state);
    assert!(!*sig);
    assert_eq!(values.data(), [4.0, 8.0, 12.0]);
}

#[test]
fn operator_receives_the_event_time() {
    // A single-port interface on both sides: the inputs tuple has one entry,
    // and the output is a bare array rather than a sequence. Binds `__op__`
    // directly instead of defining `build`.
    const STAMP: &str = r#"
class Stamp:
    def init(self, inputs):
        return None

    def reset(self, inputs, state):
        return np.zeros(1)

    def compute(self, inputs, state, instant):
        return np.array([float(instant)])

__op__ = Stamp()
"#;
    type Op = PyOperator<SignalPort<0>, ArrayPort<f64, 1>>;

    let op: Op = py_operator_source(STAMP, None);
    let pulse = ArrayView::scalar(&true);
    let mut state = Operator::init(op, pulse);
    let values = <Op as Operator>::reset(pulse, &mut state);
    assert_eq!(values.data(), [0.0]);

    // The graph's ambient event time arrives as naive nanoseconds.
    let t = Instant::from_offset(Duration::from_nanos(42));
    let values = <Op as Operator>::compute(pulse, &mut state, &t);
    assert_eq!(values.data(), [42.0]);
}

/// The whole path through the engine: a replayed source drives a Python
/// operator, whose output is recorded natively.
#[tokio::test]
async fn operator_runs_in_a_graph() {
    const DOUBLE: &str = r#"
class Double:
    def init(self, inputs):
        return None

    def reset(self, inputs, state):
        return (False, 0.0)

    def compute(self, inputs, state, instant):
        (sig, x) = inputs
        return (bool(sig), x * 2.0)

__op__ = Double()
"#;
    type In = (SignalPort<0>, ArrayPort<f64, 0>);
    type Out = (SignalPort<0>, ArrayPort<f64, 0>);

    let mut b = Builder::new(UnixTime);
    let data = Series::from_parts([], tss(&[1, 2, 3]), vec![10.0, 20.0, 30.0], 0);
    let (hs, h) = b.source(array_series(data));
    let (ds, dv) = b.op(py_operator_source::<In, Out>(DOUBLE, None), (hs, h));
    let rec = b.op(record_all(), (ds, dv));

    let mut g = b.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    let s: SeriesView<f64, 0> = g.view(rec);
    assert_eq!(s.instants(), tss(&[1, 2, 3]).as_slice());
    assert_eq!(&*s.to_contiguous(), &[20.0, 40.0, 60.0]);
}
