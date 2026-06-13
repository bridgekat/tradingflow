//! Python operators with **real NumPy** on a single embedded interpreter
//! (feature `pyflow`).
//!
//! A [`PyOperator`] is a graph node whose compute step is a Python callable. It
//! takes N `f64` array inputs and produces one `f64` array output; the operator
//! body is ordinary Python and may use NumPy/SciPy/cvxpy freely. Operators run
//! on the [`flowgraph`](flowgraph) work-stealing pool.
//!
//! This is the *only* form of Python support in the `flow` engine: Python
//! operators (Python called from Rust). There is **no** Python-as-host API
//! wrapper — graphs are built and driven from Rust, and the interpreter is
//! embedded. (The legacy `bridge` module provides the host-API wrapper for the
//! old engine; it is removed at cutover.)
//!
//! # Interpreter model (single shared interpreter; easy to switch)
//!
//! The bridge embeds **one shared CPython** and enters it per `compute` via
//! PyO3's [`Python::attach`]. This same code runs, with **no change**, on any of:
//!
//! * **Standard GIL CPython — the default.** Only one thread runs *Python* at a
//!   time, but operator work that **releases the GIL** runs truly in parallel on
//!   the pool: NumPy ufuncs/BLAS, SciPy, and convex solvers (cvxpy + CLARABEL /
//!   SCS / OSQP) all drop the GIL during their native solve. In quant operators
//!   that native/solve time *is* the cost, so solve-bound graphs parallelize;
//!   only pure-Python glue serializes. This is the default because the cvxpy
//!   solver stack has no free-threaded wheels yet.
//! * **Free-threaded CPython** (PEP 703, `python3.13t`): no GIL, so pure-Python
//!   parallelizes too. Build against it by pointing `PYO3_PYTHON` at a
//!   free-threaded venv — nothing in this module assumes one interpreter mode.
//! * **Own-GIL sub-interpreters** (PEP 684) are a possible future direction but
//!   cannot `import numpy` (NumPy declares no multiple-interpreter support), so
//!   they are not used here.
//!
//! Because operators only touch Python under `Python::attach` and never assume
//! the GIL is present or absent, switching modes is a build/runtime config change
//! (`PYO3_PYTHON` + the venv), not a code change.
//!
//! # Setup
//!
//! A venv with the operators' deps (NumPy, and SciPy/cvxpy for the optimizers):
//!
//! ```console
//! python -m venv .venv && .venv\Scripts\python -m pip install numpy scipy cvxpy
//! ```
//!
//! PyO3 links the interpreter named by `PYO3_PYTHON` at build time; the embedded
//! interpreter computes `sys.path` from the base install, so make the venv's
//! packages (and `python/`, for the `flowops` operator package) visible at
//! runtime via `PYTHONPATH`. Build and test with:
//!
//! ```console
//! set PYO3_PYTHON=<abs>\.venv\Scripts\python.exe
//! set PATH=<dir containing python3xx.dll>;%PATH%
//! set PYTHONPATH=<repo>\python;<abs>\.venv\Lib\site-packages
//! cargo test --features pyflow flow::python
//! ```
//!
//! For free-threaded instead, swap the venv for a `python3.13t` one (`py install
//! 3.13t-64`) and the dll dir accordingly. In production, point
//! `PYTHONPATH`/`PYTHONHOME` at the deployment environment.
//!
//! # Writing operators
//!
//! `source` is a Python **expression that evaluates to a callable**. `np` and
//! `numpy` are pre-injected into each operator's globals (each operator gets its
//! own globals dict — no shared mutable state between operators), so the
//! expression can use NumPy directly. There are two contracts:
//!
//! * [`PyOperator::new`] — **return mode** (ergonomic default). The callable
//!   takes `n` input ndarrays and returns a 1-D `float64` array of length
//!   `out_len`, which is copied into the output:
//!   ```python
//!   lambda a, b: a + b                       # elementwise
//!   lambda a:    np.cumsum(a)                 # one input
//!   lambda a, b: np.where(a > 0, a, b)        # branching
//!   ```
//! * [`PyOperator::writing`] — **write mode** (zero-copy output). The callable
//!   takes a writable `out` ndarray (aliasing the output buffer) followed by the
//!   inputs, and writes results into `out` in place; its return value is ignored:
//!   ```python
//!   lambda out, a, b: np.add(a, b, out=out)   # numpy `out=` writes Rust memory
//!   lambda out, a:    out.__setitem__(slice(None), np.sqrt(a))
//!   ```
//!
//! Operators must be expressions; for multi-statement logic, wrap a factory:
//! `"(lambda f: f)(lambda a: ...)"`, or define the function elsewhere and pass a
//! reference. A raised Python exception (or a wrong-length / non-`float64`
//! return) aborts the current `stabilize` with the traceback printed.
//!
//! # Building graphs
//!
//! High level, via [`Scenario`](crate::Scenario):
//! ```ignore
//! let a = sc.add_source(/* ... */, Array::zeros(&[n]));
//! let b = sc.add_source(/* ... */, Array::zeros(&[n]));
//! let sum = sc.add_py_operator("lambda a, b: a + b", &[a, b], n);          // return mode
//! let dbl = sc.add_py_operator_writing("lambda out, a: np.multiply(a, 2.0, out=out)", &[a], n);
//! ```
//! Low level, via [`GraphBuilder`](flowgraph::typed::GraphBuilder) — the inputs
//! are a `RefPorts<Array<f64>>` group, wired with a slice of handles:
//! ```ignore
//! let out = b.push(PyOperator::new("lambda a, b: a + b", n), &[*a, *bb][..]);
//! ```
//!
//! # Data model — copy-in inputs, zero-copy output
//!
//! Only Rust-owned `Array<f64>` cross node boundaries; no Python object crosses
//! an edge. The marshaling rule follows what is *address-stable for the graph's
//! lifetime*:
//!
//! * **Output is zero-copy.** A `PyOperator` allocates its output buffer once on
//!   the `init == true` build call (see [`Operator::compute`]) and only ever
//!   writes it in place, so its address is invariant for the graph's life. In
//!   write mode it is wrapped — without copying — as a writable `float64`
//!   ndarray (via [`PyArray1::borrow_from_array`] with a `None` base), so the
//!   operator writes straight into Rust memory.
//! * **Inputs are copied.** An input array is owned by an *upstream* node whose
//!   allocation is NOT graph-invariant (an upstream `Map`/`Apply` reassigns its
//!   `Array` each tick, freeing the old buffer). Each input is therefore copied
//!   into a fresh, NumPy-owned ndarray ([`PyArray1::from_slice`], one bulk
//!   memcpy). A view the operator retains then reads that owned snapshot — never
//!   freed Rust heap.
//!
//! All arrays are 1-D `float64` (the flat buffer of the cell's `Array<f64>`).
//!
//! # Safety & retention contract
//!
//! This module uses PyO3's safe API; the *entire* `unsafe` surface is the one
//! [`PyArray1::borrow_from_array`] call for the zero-copy output. Operator state
//! holds a [`Py<PyAny>`] callable, which is `Send + Sync`, so `PyOpState` is
//! `Send + Sync` without `unsafe`. The scheduler runs a given node's `compute` at
//! most once at a time, distinct nodes hold distinct callables, and free-threaded
//! CPython makes concurrent interpreter access memory-safe.
//!
//! Memory safety of the zero-copy output rests on the output buffer being
//! address-stable for the graph's life and single-writer (this node). The
//! `borrow_from_array` call uses a `None` base, so the returned array keeps
//! *nothing* alive — the state-owned buffer is freed on `Graph` drop. Operator
//! code must honour the retention contract:
//!
//! * **Inputs** are NumPy-owned copies, so retaining an input (or any slice of
//!   it) across calls is always memory-safe — it reads a live snapshot.
//! * The write-mode **`out`** array (and any view derived from it) aliases the
//!   graph-owned buffer directly. Retaining it *within* the graph's lifetime is
//!   memory-safe but a logic error (it would scribble on a later tick's output).
//!   Retaining it *past* the graph's lifetime is **undefined behavior**: the
//!   buffer is freed on `Graph` drop, so a stashed `out` array (left in the
//!   operator's globals, `sys.modules`, a thread, …) then dangles. **A write-mode
//!   operator must not let `out` escape the call.**
//!
//! # Limitations
//!
//! * 1-D `float64` only (the cell's flat buffer); no shape/dtype negotiation and
//!   no `Series` inputs yet.
//! * Return-mode operators must return a 1-D `float64` ndarray of length
//!   `out_len` (wrong length / dtype is an error).
//! * The embedded interpreter must be able to import the operators' dependencies
//!   (see *Setup*); a missing NumPy surfaces as `ModuleNotFoundError`.
//! * `source` must be a single expression evaluating to a callable.

use std::ffi::CString;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use numpy::ndarray::aview_mut1;
use numpy::{PyArray1, PyReadonlyArray1};

use flowgraph::typed::{Operator, RefPort, RefPorts};

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

/// Operator state: the config consumed from the [`PyOperator`], the compiled
/// callable (with `np` in its globals; compiled on the `init` build call), and
/// the output buffer.
pub struct PyOpState {
    source: String,
    out_len: usize,
    writes_output: bool,
    callable: Option<Py<PyAny>>,
    out: Array<f64>,
}

impl Operator for PyOperator {
    type Inputs = RefPorts<Array<f64>>;
    type Outputs = RefPort<Array<f64>>;
    type State = PyOpState;

    fn init(self) -> PyOpState {
        PyOpState {
            source: self.source,
            out_len: self.out_len,
            writes_output: self.writes_output,
            callable: None,
            out: Array::zeros(&[0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, values): (&'a [bool], &'a [&'a Array<f64>]),
        state: &'b mut PyOpState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        if init {
            // Build call: compile/evaluate the callable and size the output
            // buffer. No per-tick Python compute runs here.
            let code = CString::new(state.source.as_str()).expect("python source has interior NUL");
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
            state.callable = Some(callable);
            state.out = Array::zeros(&[state.out_len]);
            return (false, &state.out);
        }

        let n = values.len();
        let PyOpState {
            callable,
            writes_output,
            out,
            ..
        } = state;
        let callable = callable
            .as_ref()
            .expect("python callable is compiled on the build call");
        let writes = *writes_output;

        let ok = Python::attach(|py| {
            let callable = callable.bind(py);

            let mut run = || -> PyResult<()> {
                if writes {
                    // Zero-copy: wrap the (graph-invariant) output buffer as a
                    // writable ndarray with a `None` base. `view` holds the
                    // pointer with write provenance; `out` is not touched again
                    // in this branch, so the alias stays valid for the call.
                    let view = aview_mut1(out.as_mut_slice());
                    let out_arr =
                        unsafe { PyArray1::borrow_from_array(&view, py.None().into_bound(py)) };

                    let mut args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(n + 1);
                    args.push(out_arr.into_any());
                    for i in 0..n {
                        args.push(PyArray1::from_slice(py, values[i].as_slice()).into_any());
                    }
                    callable.call1(PyTuple::new(py, args)?)?; // result ignored
                    Ok(())
                } else {
                    let mut args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(n);
                    for i in 0..n {
                        args.push(PyArray1::from_slice(py, values[i].as_slice()).into_any());
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
                    let dst = out.as_mut_slice();
                    if s.len() != dst.len() {
                        return Err(PyValueError::new_err(format!(
                            "python operator returned {} elements, expected {}",
                            s.len(),
                            dst.len()
                        )));
                    }
                    dst.copy_from_slice(s);
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
        (true, &*out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a Array<f64>]),
        state: &'b PyOpState,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.out)
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
    use flowgraph::typed::{Graph, GraphBuilder, RefSource};

    use crate::Array;
    use crate::operators::Map;

    /// Return mode: element-wise NumPy add over two inputs.
    #[test]
    fn numpy_return_mode() {
        let mut b = GraphBuilder::new();
        let a = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0])));
        let bb = b.push_source(RefSource::new(Array::from_vec(&[3], vec![10.0_f64, 20.0, 30.0])));
        let out = b.push(PyOperator::new("lambda a, b: a + b", 3), &[*a, *bb][..]);
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.state_mut(a) = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        *g.state_mut(bb) = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[11.0, 22.0, 33.0]);
    }

    /// Write mode: `np.multiply(..., out=out)` writes straight into the Rust
    /// output buffer (zero-copy output).
    #[test]
    fn numpy_write_mode_zero_copy() {
        let mut b = GraphBuilder::new();
        let a = b.push_source(RefSource::new(Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0])));
        let out = b.push(
            PyOperator::writing("lambda out, a: np.multiply(a, 2.0, out=out)", 3),
            &[*a][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.state_mut(a) = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[2.0, 4.0, 6.0]);
    }

    /// Inputs arrive as real `float64` ndarrays.
    #[test]
    fn numpy_input_is_ndarray() {
        let mut b = GraphBuilder::new();
        let a = b.push_source(RefSource::new(Array::from_vec(&[2], vec![1.0_f64, 2.0])));
        let out = b.push(
            PyOperator::writing(
                "lambda out, a: out.__setitem__(0, float(\
                   type(a).__name__ == 'ndarray' and a.dtype == np.float64 and a.ndim == 1 and len(a) == 2))",
                1,
            ),
            &[*a][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.state_mut(a) = Array::from_vec(&[2], vec![1.0, 2.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[1.0]);
    }

    /// Safety regression: an operator that stashes an input across generations,
    /// downstream of a `Map` that REALLOCATES its `Array<f64>` each tick. Inputs
    /// are copied into NumPy-owned arrays, so the stash reads a live snapshot
    /// (the gen-1 value) on gen 2 — never freed Rust heap.
    #[test]
    fn retained_input_is_snapshot_not_uaf() {
        let mut b = GraphBuilder::new();
        let src = b.push_source(RefSource::new(Array::from_vec(&[1], vec![5.0_f64])));
        let mapped = b.push(Map::new(|a: &Array<f64>| a.clone()), *src);
        let out = b.push(
            PyOperator::writing(
                "lambda out, a: out.__setitem__(0, globals().setdefault('_S', a)[0])",
                1,
            ),
            &[mapped][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.state_mut(src) = Array::from_vec(&[1], vec![5.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[5.0]);

        // Map reallocates (old Box freed); the stashed snapshot still reads 5.0.
        *g.state_mut(src) = Array::from_vec(&[1], vec![99.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[5.0]);
    }

    /// Operators whose work releases the GIL run in parallel on the pool — on a
    /// **GIL** build (NumPy ufuncs / BLAS / solver calls drop the GIL) as well as
    /// free-threaded. (Pure-Python-bound operators only parallelize free-threaded;
    /// this is the GIL-releasing case the engine relies on for NumPy/cvxpy work.)
    /// The heavy body is a single-threaded NumPy ufunc loop (GIL-released, not
    /// BLAS-multithreaded, so the outer pool parallelism is the speedup measured).
    #[test]
    fn operators_run_in_parallel() {
        const K: usize = 4;
        let heavy = "lambda out, xs: out.__setitem__(0, \
                      sum(float(np.sin(np.arange(1, 1500000, dtype=np.float64) * 1.0000001).sum()) \
                      for _ in range(6)) * 0 + xs[0])";

        let mut b = GraphBuilder::new();
        let src = b.push_source(RefSource::new(Array::from_vec(&[1], vec![0.0_f64])));
        let outs: Vec<_> = (0..K)
            .map(|_| b.push(PyOperator::writing(heavy, 1), &[*src][..]))
            .collect();
        let mut g = Graph::from_builder(b);

        let mut serial = Pool::new(0);
        let mut parallel = Pool::new(K);

        *g.state_mut(src) = Array::from_vec(&[1], vec![1.0]);
        g.stabilize(&mut parallel); // warm up

        *g.state_mut(src) = Array::from_vec(&[1], vec![2.0]);
        let t = std::time::Instant::now();
        g.stabilize(&mut serial);
        let t_serial = t.elapsed();

        *g.state_mut(src) = Array::from_vec(&[1], vec![3.0]);
        let t = std::time::Instant::now();
        g.stabilize(&mut parallel);
        let t_parallel = t.elapsed();

        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let speedup = t_serial.as_secs_f64() / t_parallel.as_secs_f64();
        eprintln!(
            "{K} GIL-releasing NumPy ops, cores={cores}: serial={t_serial:?} \
             parallel={t_parallel:?} speedup={speedup:.2}x"
        );

        for &o in &outs {
            assert_eq!(g.ref_view(o).as_slice(), &[3.0]);
        }
        assert!(
            speedup > 1.5,
            "expected parallel speedup from GIL-releasing operators, got {speedup:.2}x"
        );
    }
}
