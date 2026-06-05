//! Python operators with **true parallelism** via PEP 684 own-GIL
//! sub-interpreters (feature `pyflow`).
//!
//! Each [`PyOperator`] carries its own Python sub-interpreter as operator
//! state. Because every sub-interpreter has its own GIL, K Python operator
//! nodes run genuinely in parallel on the work-stealing pool — even pure-Python
//! work — rather than serializing on one global GIL.
//!
//! Data model (the flowgraph "Rust crosses edges, Python lives in state"
//! design): only Rust-owned `Array<f64>` cross node boundaries. On each
//! `compute` the input arrays are marshaled into Python lists, the operator's
//! callable runs inside its sub-interpreter, and the returned list is written
//! back into the Rust output array. No Python object ever crosses an edge.
//! Marshaling uses only core CPython (`PyList`/`PyFloat`/`PySequence`), which is
//! robust inside sub-interpreters (no C-extension required).
//!
//! Only Python *operators* are supported — there is no Python-as-host API
//! wrapper. Graphs are built and driven from Rust; Python is embedded.
//!
//! # Safety
//!
//! The sub-interpreter lifecycle is hand-written `unsafe` FFI (PyO3's safe API
//! targets the main interpreter only). `PyOpState` is `Send + Sync` by hand:
//! the scheduler runs a given node's `compute` at most once at a time, and the
//! interpreter / callable are only ever touched while holding *this*
//! interpreter's own GIL, on whatever worker thread runs the node.

use std::ffi::CString;

use pyo3::ffi;
use pyo3::prelude::*;

use flowgraph::typed::{Port, SliceNotify, SliceRefs};

use super::op::Operator;
use crate::Array;

// ===========================================================================
// SubInterp — an own-GIL sub-interpreter (PEP 684)
// ===========================================================================

struct SubInterp(*mut ffi::PyInterpreterState);

// SAFETY: see module-level docs — single-writer-per-generation, accessed only
// under this interpreter's own GIL.
unsafe impl Send for SubInterp {}
unsafe impl Sync for SubInterp {}

impl SubInterp {
    /// Create a fresh own-GIL sub-interpreter. Needs the main GIL, so it
    /// brackets the work in `Python::attach`; runs at graph-build time.
    fn new() -> Self {
        Python::attach(|_| unsafe {
            let main = ffi::PyThreadState_Get();
            let config = ffi::PyInterpreterConfig {
                use_main_obmalloc: 0,
                allow_fork: 0,
                allow_exec: 0,
                allow_threads: 1,
                allow_daemon_threads: 0,
                check_multi_interp_extensions: 1,
                gil: ffi::PyInterpreterConfig_OWN_GIL,
            };
            let mut sub = std::ptr::null_mut();
            let status = ffi::Py_NewInterpreterFromConfig(&mut sub, &config);
            assert!(
                ffi::PyStatus_Exception(status) == 0,
                "sub-interpreter creation failed"
            );
            let interp = ffi::PyInterpreterState_Get(); // `sub` is current here
            // Drop the creation thread-state; restore the main thread-state.
            ffi::PyThreadState_Clear(sub);
            ffi::PyThreadState_DeleteCurrent();
            ffi::PyEval_RestoreThread(main);
            SubInterp(interp)
        })
    }
}

impl Drop for SubInterp {
    fn drop(&mut self) {
        unsafe {
            let ts = ffi::PyThreadState_New(self.0);
            ffi::PyEval_RestoreThread(ts);
            ffi::Py_EndInterpreter(ts);
        }
    }
}

/// Enter `interp` on the current thread (acquiring its own GIL), run `f`, then
/// leave (releasing the GIL). A fresh thread-state per call — the setup is
/// microseconds and CPython forbids stray thread-states at `Py_EndInterpreter`.
///
/// `f` must not panic: it runs while the GIL is held, so a panic would leak the
/// thread-state and never release the GIL. The callers below return a status
/// and panic *after* leaving instead.
///
/// # Safety
/// `interp` must be a live interpreter created by [`SubInterp::new`].
unsafe fn with_interp<R>(interp: *mut ffi::PyInterpreterState, f: impl FnOnce() -> R) -> R {
    unsafe {
        let ts = ffi::PyThreadState_New(interp); // no GIL required
        ffi::PyEval_RestoreThread(ts); // acquire this interpreter's GIL
        let r = f();
        ffi::PyThreadState_Clear(ts);
        ffi::PyThreadState_DeleteCurrent(); // delete ts, release the GIL
        r
    }
}

// ===========================================================================
// Marshaling (called only while holding the relevant interpreter's GIL)
// ===========================================================================

/// Build a Python `list[float]` from a Rust slice. Returns a new reference.
unsafe fn slice_to_pylist(xs: &[f64]) -> *mut ffi::PyObject {
    unsafe {
        let list = ffi::PyList_New(xs.len() as ffi::Py_ssize_t);
        for (i, &x) in xs.iter().enumerate() {
            // PyList_SetItem steals the float reference.
            ffi::PyList_SetItem(list, i as ffi::Py_ssize_t, ffi::PyFloat_FromDouble(x));
        }
        list
    }
}

/// Write a Python sequence of floats into `out`. Returns `false` on length
/// mismatch or a non-numeric element (clearing the Python error).
unsafe fn pyseq_to_slice(obj: *mut ffi::PyObject, out: &mut [f64]) -> bool {
    unsafe {
        let len = ffi::PyObject_Length(obj);
        if len < 0 {
            ffi::PyErr_Clear();
            return false;
        }
        if len as usize != out.len() {
            return false;
        }
        for (i, slot) in out.iter_mut().enumerate() {
            let item = ffi::PySequence_GetItem(obj, i as ffi::Py_ssize_t); // new ref
            if item.is_null() {
                ffi::PyErr_Clear();
                return false;
            }
            let v = ffi::PyFloat_AsDouble(item);
            ffi::Py_DECREF(item);
            if v == -1.0 && !ffi::PyErr_Occurred().is_null() {
                ffi::PyErr_Clear();
                return false;
            }
            *slot = v;
        }
        true
    }
}

// ===========================================================================
// PyOperator — N f64-array inputs → 1 f64-array output, via a Python callable
// ===========================================================================

/// A Python-backed operator. `source` is a Python expression that evaluates to
/// a callable taking `n` positional `list[float]` arguments (one per input) and
/// returning a `list[float]` of length `out_len`, e.g.
/// `"lambda a, b: [x + y for x, y in zip(a, b)]"`.
pub struct PyOperator {
    source: String,
    out_len: usize,
}

impl PyOperator {
    pub fn new(source: impl Into<String>, out_len: usize) -> Self {
        Self {
            source: source.into(),
            out_len,
        }
    }
}

/// Operator state: the sub-interpreter and the compiled callable (a new
/// reference owned by that interpreter).
pub struct PyOpState {
    interp: SubInterp,
    callable: *mut ffi::PyObject,
}

// SAFETY: see module docs.
unsafe impl Send for PyOpState {}
unsafe impl Sync for PyOpState {}

impl Drop for PyOpState {
    fn drop(&mut self) {
        // Release the callable inside its interpreter *before* `interp` drops
        // (field order: this body runs first, then `interp`'s `Drop`).
        unsafe {
            with_interp(self.interp.0, || {
                ffi::Py_DECREF(self.callable);
            });
        }
    }
}

impl Operator for PyOperator {
    type Inputs = [Port<Array<f64>>];
    type Output = Array<f64>;
    type State = PyOpState;

    fn init(&self, _inputs: SliceRefs<'_, Port<Array<f64>>>) -> (PyOpState, Array<f64>) {
        let interp = SubInterp::new();
        let source = CString::new(self.source.as_str()).expect("python source has interior NUL");
        let callable = unsafe {
            with_interp(interp.0, || {
                // Evaluate the source as an expression in this interpreter's
                // own `__main__` namespace → the callable (new reference).
                let main_mod = ffi::PyImport_AddModule(c"__main__".as_ptr()); // borrowed
                assert!(!main_mod.is_null(), "no __main__ in sub-interpreter");
                let globals = ffi::PyModule_GetDict(main_mod); // borrowed
                let callable =
                    ffi::PyRun_String(source.as_ptr(), ffi::Py_eval_input, globals, globals);
                if callable.is_null() {
                    ffi::PyErr_Print();
                    panic!("python operator source failed to compile/evaluate");
                }
                callable
            })
        };
        (
            PyOpState { interp, callable },
            Array::zeros(&[self.out_len]),
        )
    }

    fn compute(
        state: &mut PyOpState,
        inputs: SliceRefs<'_, Port<Array<f64>>>,
        output: &mut Array<f64>,
        _produced: SliceNotify<'_, Port<Array<f64>>>,
    ) -> bool {
        let n = inputs.len();
        let callable = state.callable;
        let ok = unsafe {
            with_interp(state.interp.0, || {
                // args = tuple(list(input_i) for i in 0..n)
                let args = ffi::PyTuple_New(n as ffi::Py_ssize_t);
                for i in 0..n {
                    let list = slice_to_pylist(inputs.get(i).as_slice());
                    ffi::PyTuple_SetItem(args, i as ffi::Py_ssize_t, list); // steals ref
                }
                let res = ffi::PyObject_CallObject(callable, args);
                ffi::Py_DECREF(args);
                if res.is_null() {
                    ffi::PyErr_Print();
                    return false;
                }
                let ok = pyseq_to_slice(res, output.as_mut_slice());
                ffi::Py_DECREF(res);
                ok
            })
        };
        assert!(
            ok,
            "python operator raised, or returned a non-list / wrong-length result"
        );
        true
    }
}

// ===========================================================================
// Tests (run with `--features pyflow`; need libpython linkable + on PATH)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::PyOperator;
    use flowgraph::core::Pool;
    use flowgraph::typed::{Graph, GraphBuilder};

    use crate::Array;
    use crate::flow::{Adapt, Const};

    /// Correctness: a two-input element-wise Python sum.
    #[test]
    fn py_operator_elementwise_add() {
        // `auto-initialize` initializes Python on the first `Python::attach`
        // (inside `SubInterp::new`); no explicit init needed.
        let mut b = GraphBuilder::new();
        let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]))), ());
        let bb = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![10.0_f64, 20.0, 30.0]))), ());
        let out = b.push(
            Adapt::new(PyOperator::new(
                "lambda a, b: [x + y for x, y in zip(a, b)]",
                3,
            )),
            &[a, bb][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        *g.cell_mut(bb) = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[11.0, 22.0, 33.0]);
    }

    /// True parallelism: K CPU-bound pure-Python operators, each in its own
    /// sub-interpreter, run far faster on a K-worker pool than on a serial one.
    #[test]
    fn py_operators_run_in_parallel() {
        const K: usize = 4;
        // Heavy pure-Python loop; returns [xs[0]] (the loop's value is discarded
        // via * 0, so output == input — easy to assert).
        let heavy = "lambda xs: [sum((i * 1103515245 + 12345) % 2147483647 \
                      for i in range(400000)) * 0 + xs[0]]";

        let mut b = GraphBuilder::new();
        let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
        let outs: Vec<_> = (0..K)
            .map(|_| b.push(Adapt::new(PyOperator::new(heavy, 1)), &[src][..]))
            .collect();
        let mut g = Graph::from_builder(b);

        let mut serial = Pool::new(0);
        let mut parallel = Pool::new(K);

        // Warm up each sub-interpreter once.
        *g.cell_mut(src) = Array::scalar(1.0);
        g.stabilize(&mut parallel);

        *g.cell_mut(src) = Array::scalar(2.0);
        let t = std::time::Instant::now();
        g.stabilize(&mut serial);
        let t_serial = t.elapsed();

        *g.cell_mut(src) = Array::scalar(3.0);
        let t = std::time::Instant::now();
        g.stabilize(&mut parallel);
        let t_parallel = t.elapsed();

        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let speedup = t_serial.as_secs_f64() / t_parallel.as_secs_f64();
        eprintln!(
            "{K} heavy pure-Python ops, cores={cores}: serial={t_serial:?} \
             parallel={t_parallel:?} speedup={speedup:.2}x"
        );

        for &o in &outs {
            assert_eq!(g.cell(o).as_slice(), &[3.0]);
        }
        // Own-GIL sub-interpreters → real parallelism. Conservative threshold.
        assert!(
            speedup > 1.7,
            "expected parallel speedup from own-GIL sub-interpreters, got {speedup:.2}x"
        );
    }
}
