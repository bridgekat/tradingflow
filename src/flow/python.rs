//! Python operators with **true parallelism** and **real NumPy**, on a single
//! free-threaded interpreter (feature `pyflow`).
//!
//! Build/run against a free-threaded CPython (`python3.13t`): point
//! `PYO3_PYTHON` at a free-threaded venv, put its `python3xxt.dll` directory on
//! `PATH`, and point the embedded interpreter at the environment that has NumPy
//! via `PYTHONPATH` (or `PYTHONHOME`). On a free-threaded build there is no
//! global GIL, so operator nodes run genuinely in parallel on the work-stealing
//! pool — pure-Python *and* NumPy — within one shared interpreter. On a standard
//! GIL-enabled build the bridge is still correct, but `compute` calls serialize
//! (no parallelism). (The earlier own-GIL sub-interpreter design was abandoned:
//! it gave pure-Python parallelism but could never `import numpy`, which declares
//! no multiple-interpreter support.)
//!
//! # Data model — copy-in inputs, zero-copy output
//!
//! Operators see real `numpy.ndarray`s. Only Rust-owned `Array<f64>` cross node
//! boundaries; the marshaling follows what is *address-stable for the graph's
//! lifetime*:
//!
//! * **Output is zero-copy.** A `PyOperator` allocates its output buffer once in
//!   [`init`](Operator::init) and only ever writes it in place, so its address
//!   is invariant for the graph's life. In *write mode* it is wrapped — without
//!   copying — as a writable 1-D `float64` ndarray (via
//!   [`PyArray1::borrow_from_array`] with a `None` base), and the operator
//!   writes results straight into Rust memory (e.g. `np.add(a, b, out=out)`).
//! * **Inputs are copied.** An input array is owned by an *upstream* node whose
//!   allocation is NOT graph-invariant (an upstream `Map`/`Apply` reassigns its
//!   `Array` each tick, freeing the old buffer). Each input is therefore copied
//!   into a fresh, NumPy-owned ndarray ([`PyArray1::from_slice`]). A view the
//!   operator retains across calls then reads that owned snapshot — never freed
//!   Rust heap. The copy is a single bulk memcpy.
//!
//! Two operator contracts (pick at construction):
//!
//! * [`PyOperator::new`] — *return mode*. `f(*inputs) -> ndarray`; the returned
//!   1-D `float64` array of length `out_len` is copied into the output. Drop-in
//!   for `lambda a, b: a + b`.
//! * [`PyOperator::writing`] — *write mode*. `f(out, *inputs)`; `out` is the
//!   zero-copy output ndarray and the operator writes into it (return ignored),
//!   e.g. `lambda out, a, b: np.add(a, b, out=out)`.
//!
//! `np` (and `numpy`) are pre-injected into each operator's globals, so a source
//! expression can use NumPy directly. Each operator gets its own globals dict
//! (no shared mutable module state between operators).
//!
//! Only Python *operators* are supported — there is no Python-as-host API
//! wrapper. Graphs are built and driven from Rust; Python is embedded.
//!
//! # Safety
//!
//! This module uses PyO3's safe API; there is no hand-written interpreter FFI.
//! Operator state is a [`Py<PyAny>`] callable, which is `Send + Sync`, so
//! `PyOpState` is `Send + Sync` without `unsafe`. The scheduler runs a given
//! node's `compute` at most once at a time; different nodes' callables are
//! distinct objects, and free-threaded CPython makes concurrent access to the
//! interpreter memory-safe.
//!
//! Memory safety of the zero-copy output rests on the output buffer being
//! address-stable for the graph's life (allocated once, written in place, never
//! replaced) and single-writer (this node). The one `unsafe` call,
//! [`PyArray1::borrow_from_array`], wraps that buffer with a `None` base, so the
//! returned array keeps *nothing* alive — the graph cell owns the buffer and
//! frees it when the `Graph` is dropped.
//!
//! Retention contract (operator code must honour it):
//! * **Inputs** are copied into NumPy-owned arrays, so retaining an input (or any
//!   slice of it) across calls is always memory-safe — it reads a live snapshot.
//! * The write-mode **`out`** array (and any view derived from it) aliases the
//!   graph-owned buffer directly. Retaining it *within* the graph's lifetime is
//!   memory-safe (it aliases stable memory) but a logic error — it would
//!   scribble on a later tick's output. Retaining it *past* the graph's lifetime
//!   is **undefined behavior**: the buffer is freed on `Graph` drop and a stashed
//!   `out` array (left in the operator's globals, `sys.modules`, a thread, …)
//!   then dangles. A write-mode operator must not let `out` escape the call.

use std::ffi::CString;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use numpy::ndarray::aview_mut1;
use numpy::{PyArray1, PyReadonlyArray1};

use flowgraph::typed::{Port, SliceNotify, SliceRefs};

use super::op::Operator;
use crate::Array;

// ===========================================================================
// PyOperator — N f64-array inputs → 1 f64-array output, via a Python callable
// ===========================================================================

/// A Python-backed operator. `source` is a Python expression that evaluates to
/// a callable; the contract depends on the construction mode (see the module
/// docs and [`new`](Self::new) / [`writing`](Self::writing)).
pub struct PyOperator {
    source: String,
    out_len: usize,
    /// `true` → write mode (`f(out, *inputs)`, writes `out` in place);
    /// `false` → return mode (`f(*inputs) -> ndarray`, landed into the output).
    writes_output: bool,
}

impl PyOperator {
    /// Return mode (ergonomic default). The callable takes `n` positional
    /// `float64` ndarrays (copies of the inputs) and returns a 1-D `float64`
    /// array of length `out_len`, e.g. `"lambda a, b: a + b"`.
    pub fn new(source: impl Into<String>, out_len: usize) -> Self {
        Self {
            source: source.into(),
            out_len,
            writes_output: false,
        }
    }

    /// Write mode (zero-copy output). The callable takes a writable `float64`
    /// ndarray `out` (aliasing the output buffer) followed by `n` input ndarrays
    /// and writes its results into `out` in place; its return value is ignored,
    /// e.g. `"lambda out, a, b: np.add(a, b, out=out)"`.
    pub fn writing(source: impl Into<String>, out_len: usize) -> Self {
        Self {
            source: source.into(),
            out_len,
            writes_output: true,
        }
    }
}

/// Operator state: the compiled callable (with `np` in its globals) and the mode.
pub struct PyOpState {
    callable: Py<PyAny>,
    writes_output: bool,
}

impl Operator for PyOperator {
    type Inputs = [Port<Array<f64>>];
    type Output = Array<f64>;
    type State = PyOpState;

    fn init(&self, _inputs: SliceRefs<'_, Port<Array<f64>>>) -> (PyOpState, Array<f64>) {
        let code = CString::new(self.source.as_str()).expect("python source has interior NUL");
        let callable: Py<PyAny> = Python::attach(|py| {
            let run = || -> PyResult<Py<PyAny>> {
                // Per-operator globals with NumPy pre-injected, so the source
                // expression can use `np` / `numpy` directly.
                let globals = PyDict::new(py);
                let np = py.import("numpy")?;
                globals.set_item("np", &np)?;
                globals.set_item("numpy", &np)?;
                let obj = py.eval(code.as_c_str(), Some(&globals), None)?;
                Ok(obj.unbind())
            };
            run().unwrap_or_else(|e| {
                e.print(py);
                panic!("python operator source failed to compile/evaluate");
            })
        });
        (
            PyOpState {
                callable,
                writes_output: self.writes_output,
            },
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
        let writes = state.writes_output;

        let ok = Python::attach(|py| {
            let callable = state.callable.bind(py);

            let mut run = || -> PyResult<()> {
                if writes {
                    // Zero-copy: wrap the (graph-invariant) output buffer as a
                    // writable ndarray with a `None` base. `view` holds the
                    // pointer with write provenance; `output` is not touched
                    // again in this branch, so the alias stays valid for the
                    // call.
                    let view = aview_mut1(output.as_mut_slice());
                    let out_arr =
                        unsafe { PyArray1::borrow_from_array(&view, py.None().into_bound(py)) };

                    let mut args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(n + 1);
                    args.push(out_arr.into_any());
                    for i in 0..n {
                        args.push(PyArray1::from_slice(py, inputs.get(i).as_slice()).into_any());
                    }
                    callable.call1(PyTuple::new(py, args)?)?; // result ignored
                    Ok(())
                } else {
                    let mut args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(n);
                    for i in 0..n {
                        args.push(PyArray1::from_slice(py, inputs.get(i).as_slice()).into_any());
                    }
                    let ret = callable.call1(PyTuple::new(py, args)?)?;
                    let arr: PyReadonlyArray1<f64> = ret.extract().map_err(|_| {
                        PyValueError::new_err(
                            "return-mode python operator must return a 1-D float64 ndarray",
                        )
                    })?;
                    let s = arr
                        .as_slice()
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let out = output.as_mut_slice();
                    if s.len() != out.len() {
                        return Err(PyValueError::new_err(format!(
                            "python operator returned {} elements, expected {}",
                            s.len(),
                            out.len()
                        )));
                    }
                    out.copy_from_slice(s);
                    Ok(())
                }
            };

            match run() {
                Ok(()) => true,
                Err(e) => {
                    e.print(py);
                    false
                }
            }
        });
        assert!(ok, "python operator failed (see traceback above)");
        true
    }
}

// ===========================================================================
// Tests — build/run against a free-threaded interpreter:
//   PYO3_PYTHON=<ft venv python>  PATH+=<dir of python3xxt.dll>
//   PYTHONPATH=<ft venv site-packages>  cargo test --features pyflow flow::python
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::PyOperator;
    use flowgraph::core::Pool;
    use flowgraph::typed::{Graph, GraphBuilder};

    use crate::Array;
    use crate::flow::{Adapt, Const, Map};

    /// Sanity: the interpreter is genuinely free-threaded (no GIL).
    #[test]
    fn interpreter_is_free_threaded() {
        use pyo3::prelude::*;
        Python::attach(|py| {
            let gil_on: bool = py
                .import("sys")
                .unwrap()
                .getattr("_is_gil_enabled")
                .unwrap()
                .call0()
                .unwrap()
                .extract()
                .unwrap();
            assert!(!gil_on, "tests must run on a free-threaded build (python3.13t)");
        });
    }

    /// Return mode: element-wise NumPy add over two inputs.
    #[test]
    fn numpy_return_mode() {
        let mut b = GraphBuilder::new();
        let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]))), ());
        let bb = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![10.0_f64, 20.0, 30.0]))), ());
        let out = b.push(
            Adapt::new(PyOperator::new("lambda a, b: a + b", 3)),
            &[a, bb][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        *g.cell_mut(bb) = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[11.0, 22.0, 33.0]);
    }

    /// Write mode: `np.multiply(..., out=out)` writes straight into the Rust
    /// output buffer (zero-copy output).
    #[test]
    fn numpy_write_mode_zero_copy() {
        let mut b = GraphBuilder::new();
        let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]))), ());
        let out = b.push(
            Adapt::new(PyOperator::writing("lambda out, a: np.multiply(a, 2.0, out=out)", 3)),
            &[a][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[2.0, 4.0, 6.0]);
    }

    /// Inputs arrive as real `float64` ndarrays.
    #[test]
    fn numpy_input_is_ndarray() {
        let mut b = GraphBuilder::new();
        let a = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![1.0_f64, 2.0]))), ());
        let out = b.push(
            Adapt::new(PyOperator::writing(
                "lambda out, a: out.__setitem__(0, float(\
                   type(a).__name__ == 'ndarray' and a.dtype == np.float64 and a.ndim == 1 and len(a) == 2))",
                1,
            )),
            &[a][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(a) = Array::from_vec(&[2], vec![1.0, 2.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[1.0]);
    }

    /// Safety regression: an operator that stashes an input across generations,
    /// downstream of a `Map` that REALLOCATES its `Array<f64>` each tick. Inputs
    /// are copied into NumPy-owned arrays, so the stash reads a live snapshot
    /// (the gen-1 value) on gen 2 — never freed Rust heap.
    #[test]
    fn retained_input_is_snapshot_not_uaf() {
        let mut b = GraphBuilder::new();
        let src = b.push(Adapt::new(Const(Array::from_vec(&[1], vec![5.0_f64]))), ());
        let mapped = b.push(Adapt::new(Map::new(|a: &Array<f64>| a.clone())), src);
        let out = b.push(
            Adapt::new(PyOperator::writing(
                "lambda out, a: out.__setitem__(0, globals().setdefault('_S', a)[0])",
                1,
            )),
            &[mapped][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(src) = Array::from_vec(&[1], vec![5.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[5.0]);

        // Map reallocates (old Box freed); the stashed snapshot still reads 5.0.
        *g.cell_mut(src) = Array::from_vec(&[1], vec![99.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[5.0]);
    }

    /// True parallelism on the free-threaded build: K CPU-bound pure-Python
    /// operators run far faster on a K-worker pool than on a serial one (no GIL).
    #[test]
    fn operators_run_in_parallel() {
        const K: usize = 4;
        let heavy = "lambda out, xs: out.__setitem__(0, \
                      sum((i * 1103515245 + 12345) % 2147483647 \
                      for i in range(400000)) * 0 + xs[0])";

        let mut b = GraphBuilder::new();
        let src = b.push(Adapt::new(Const(Array::from_vec(&[1], vec![0.0_f64]))), ());
        let outs: Vec<_> = (0..K)
            .map(|_| b.push(Adapt::new(PyOperator::writing(heavy, 1)), &[src][..]))
            .collect();
        let mut g = Graph::from_builder(b);

        let mut serial = Pool::new(0);
        let mut parallel = Pool::new(K);

        *g.cell_mut(src) = Array::from_vec(&[1], vec![1.0]);
        g.stabilize(&mut parallel); // warm up

        *g.cell_mut(src) = Array::from_vec(&[1], vec![2.0]);
        let t = std::time::Instant::now();
        g.stabilize(&mut serial);
        let t_serial = t.elapsed();

        *g.cell_mut(src) = Array::from_vec(&[1], vec![3.0]);
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
        assert!(
            speedup > 1.7,
            "expected GIL-free parallelism, got {speedup:.2}x"
        );
    }
}
