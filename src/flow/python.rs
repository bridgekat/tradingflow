//! Python operators with **true parallelism** via PEP 684 own-GIL
//! sub-interpreters (feature `pyflow`).
//!
//! Each [`PyOperator`] carries its own Python sub-interpreter as operator
//! state. Because every sub-interpreter has its own GIL, K Python operator
//! nodes run genuinely in parallel on the work-stealing pool — even pure-Python
//! work — rather than serializing on one global GIL.
//!
//! # Data model — copy-in inputs, zero-copy output
//!
//! Only Rust-owned `Array<f64>` cross node boundaries; no Python object ever
//! crosses an edge. The marshaling rule follows what is *address-stable for the
//! graph's lifetime*:
//!
//! * **Output is zero-copy.** A `PyOperator` allocates its output buffer once in
//!   [`init`](Operator::init) and only ever writes it in place, so that buffer's
//!   address is invariant for the graph's life. In *write mode* it is exposed to
//!   Python as a writable `memoryview` and the operator writes results straight
//!   into Rust memory.
//! * **Inputs are copied.** An input array is owned by an *upstream* node, whose
//!   allocation is NOT graph-invariant: an upstream `Map`/`Apply` reassigns its
//!   `Array` each tick (`*output = …`), freeing the old `Box<[f64]>`. Handing
//!   Python a zero-copy view over that buffer would dangle the moment the
//!   operator stashed (even a slice of) it across calls and the upstream
//!   reallocated. So each input is copied into an immutable, GC-owned `bytes`
//!   snapshot and exposed as a read-only f64 `memoryview` over *that*. A view the
//!   operator retains across calls then reads a live Python snapshot — never
//!   freed Rust heap. The copy is a single bulk memcpy (no per-element boxing,
//!   8 B/elem), far cheaper than the legacy list-of-floats marshaling.
//!
//! Why not NumPy: NumPy's C extension declares no multiple-interpreter support,
//! so `import numpy` fails inside an own-GIL sub-interpreter (the design that
//! buys us parallelism). The buffer protocol (`memoryview`) is pure CPython,
//! works in sub-interpreters, and gives the same zero-copy guarantee for the
//! output.
//!
//! Two operator contracts (pick at construction):
//!
//! * [`PyOperator::new`] — *return mode*. `f(*inputs) -> sequence`; each input is
//!   a read-only f64 memoryview, the returned `out_len` floats are written into
//!   the output. Drop-in for `lambda a, b: [x + y for x, y in zip(a, b)]`.
//! * [`PyOperator::writing`] — *write mode*. `f(out, *inputs)`; `out` is a
//!   writable f64 memoryview over the (zero-copy) output buffer and the inputs
//!   are read-only ones. The operator writes into `out`; its return is ignored.
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
//!
//! Memory safety of the marshaling rests on two facts, not on any "don't retain
//! views" promise from operator code:
//!
//! * Input views alias an immutable GC-owned `bytes` copy, so a view (or any
//!   child slice of it) the operator stashes stays valid as long as Python holds
//!   it — it can never dangle into reused Rust heap.
//! * The output view aliases the operator's own output buffer, whose address is
//!   invariant for the graph's life (allocated once, written in place, never
//!   replaced) and which has exactly one writer (this node). A retained child of
//!   it therefore aliases still-valid memory.
//!
//! Retaining or deriving a view past the call is thus *memory-safe* but a logic
//! error: an input view reads its creation-time snapshot, and a stashed writable
//! output view can scribble on a later tick's output. It is a correctness
//! contract, not a soundness one.
//!
//! [`with_interp`] wraps the GIL-held closure in `catch_unwind`: a panic
//! unwinding through CPython's C frames is UB and would leak the GIL and wedge
//! the interpreter, so panics are caught, the GIL is always released, and the
//! payload is re-raised after leaving the interpreter.

use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use pyo3::ffi;
use pyo3::prelude::*;

use flowgraph::typed::{Port, SliceNotify, SliceRefs};

use super::op::Operator;
use crate::Array;

const F64_SIZE: ffi::Py_ssize_t = std::mem::size_of::<f64>() as ffi::Py_ssize_t;

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
/// The closure is run under `catch_unwind`: a panic must not unwind through the
/// CPython C frames above it (UB) and must not skip releasing the GIL. On panic
/// the GIL is still released and the thread-state deleted, then the payload is
/// re-raised on this thread *after* leaving the interpreter (the pool re-raises
/// it on the driver).
///
/// # Safety
/// `interp` must be a live interpreter created by [`SubInterp::new`].
unsafe fn with_interp<R>(interp: *mut ffi::PyInterpreterState, f: impl FnOnce() -> R) -> R {
    unsafe {
        let ts = ffi::PyThreadState_New(interp); // no GIL required
        assert!(!ts.is_null(), "PyThreadState_New failed (out of memory)");
        ffi::PyEval_RestoreThread(ts); // acquire this interpreter's GIL
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        ffi::PyThreadState_Clear(ts);
        ffi::PyThreadState_DeleteCurrent(); // delete ts, release the GIL
        match r {
            Ok(v) => v,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}

// ===========================================================================
// Marshaling (called only while holding the relevant interpreter's GIL)
// ===========================================================================

/// Wrap a contiguous `f64` buffer as a 1-D writable `memoryview` of format `"d"`,
/// **zero-copy** (the view's `Py_buffer.obj` is NULL — it aliases `ptr`
/// directly). Returns a new reference, or null on failure (Python error set).
///
/// Use ONLY for buffers whose address is invariant for the graph's life (the
/// operator's own output). Returned views must not outlive that buffer.
///
/// # Safety
/// `ptr` must point to `len` valid, contiguous, graph-invariant `f64`s.
unsafe fn f64_output_view(ptr: *mut f64, len: usize) -> *mut ffi::PyObject {
    unsafe {
        // `format` must outlive the view; a `c"d"` literal is 'static.
        let mut shape = [len as ffi::Py_ssize_t];
        let mut strides = [F64_SIZE];
        let mut view: ffi::Py_buffer = std::mem::zeroed();
        view.buf = ptr as *mut c_void;
        view.obj = std::ptr::null_mut();
        view.len = len as ffi::Py_ssize_t * F64_SIZE;
        view.itemsize = F64_SIZE;
        view.readonly = 0;
        view.ndim = 1;
        view.format = c"d".as_ptr() as *mut c_char;
        view.shape = shape.as_mut_ptr();
        view.strides = strides.as_mut_ptr();
        view.suboffsets = std::ptr::null_mut();
        view.internal = std::ptr::null_mut();
        // `PyMemoryView_FromBuffer` copies `view` (incl. shape/strides *values*)
        // into the memoryview, so the stack arrays need only live for this call.
        ffi::PyMemoryView_FromBuffer(&view)
    }
}

/// Copy a contiguous `f64` slice into an immutable, GC-owned `bytes` snapshot and
/// return a **read-only** f64 `memoryview` over it (format `"d"`). The view (and
/// any slice the operator derives) keeps the `bytes` alive, so it can never
/// dangle into reused Rust heap. Returns a new reference, or null on failure.
///
/// # Safety
/// `ptr` must point to `len` valid, contiguous `f64`s for the duration of this
/// call (the data is copied out before returning).
unsafe fn f64_input_view(ptr: *const f64, len: usize) -> *mut ffi::PyObject {
    unsafe {
        let nbytes = len as ffi::Py_ssize_t * F64_SIZE;
        // Bulk memcpy into immutable, GC-owned bytes.
        let bytes = ffi::PyBytes_FromStringAndSize(ptr as *const c_char, nbytes);
        if bytes.is_null() {
            return std::ptr::null_mut();
        }
        // A memoryview *over the object* holds a reference to `bytes`, so the
        // snapshot lives as long as the view (or any child slice of it).
        let mv_b = ffi::PyMemoryView_FromObject(bytes);
        ffi::Py_DECREF(bytes);
        if mv_b.is_null() {
            return std::ptr::null_mut();
        }
        // Reinterpret the 'B' view as f64: `mv_b.cast('d')` (shares the buffer,
        // keeps `bytes` alive, stays read-only).
        let cast = ffi::PyObject_GetAttrString(mv_b, c"cast".as_ptr());
        if cast.is_null() {
            ffi::Py_DECREF(mv_b);
            return std::ptr::null_mut();
        }
        let fmt = ffi::PyUnicode_FromString(c"d".as_ptr());
        let cast_args = ffi::PyTuple_New(1);
        if fmt.is_null() || cast_args.is_null() {
            // OOM: clean up still-owned refs (fmt not yet stolen) and bail.
            ffi::Py_XDECREF(fmt);
            ffi::Py_XDECREF(cast_args);
            ffi::Py_DECREF(cast);
            ffi::Py_DECREF(mv_b);
            return std::ptr::null_mut();
        }
        ffi::PyTuple_SetItem(cast_args, 0, fmt); // steals `fmt`
        let mv_d = ffi::PyObject_CallObject(cast, cast_args);
        ffi::Py_DECREF(cast);
        ffi::Py_DECREF(cast_args);
        ffi::Py_DECREF(mv_b);
        mv_d
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
/// a callable; the contract depends on the construction mode (see the module
/// docs and [`new`](Self::new) / [`writing`](Self::writing)).
pub struct PyOperator {
    source: String,
    out_len: usize,
    /// `true` → write mode (`f(out, *inputs)`, writes `out` in place);
    /// `false` → return mode (`f(*inputs) -> sequence`, landed into the output).
    writes_output: bool,
}

impl PyOperator {
    /// Return mode (ergonomic default). The callable takes `n` positional
    /// read-only f64 `memoryview`s (one per input) and returns a `sequence` of
    /// `out_len` floats, e.g. `"lambda a, b: [x + y for x, y in zip(a, b)]"`.
    pub fn new(source: impl Into<String>, out_len: usize) -> Self {
        Self {
            source: source.into(),
            out_len,
            writes_output: false,
        }
    }

    /// Write mode (zero-copy output). The callable takes a writable f64
    /// `memoryview` `out` followed by `n` read-only f64 `memoryview`s, and writes
    /// its `out_len` results into `out` in place; its return value is ignored,
    /// e.g. `"lambda out, a, b: [out.__setitem__(i, a[i] + b[i]) for i in range(len(out))]"`.
    pub fn writing(source: impl Into<String>, out_len: usize) -> Self {
        Self {
            source: source.into(),
            out_len,
            writes_output: true,
        }
    }
}

/// Operator state: the sub-interpreter and the compiled callable (a new
/// reference owned by that interpreter).
pub struct PyOpState {
    interp: SubInterp,
    callable: *mut ffi::PyObject,
    writes_output: bool,
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
        // The closure returns the callable (or null); we validate *after*
        // leaving the interpreter so the assert never panics with the GIL held.
        let callable = unsafe {
            with_interp(interp.0, || {
                // Evaluate the source as an expression in this interpreter's
                // own `__main__` namespace → the callable (new reference).
                let main_mod = ffi::PyImport_AddModule(c"__main__".as_ptr()); // borrowed
                if main_mod.is_null() {
                    return std::ptr::null_mut();
                }
                let globals = ffi::PyModule_GetDict(main_mod); // borrowed
                let callable =
                    ffi::PyRun_String(source.as_ptr(), ffi::Py_eval_input, globals, globals);
                if callable.is_null() {
                    ffi::PyErr_Print();
                }
                callable
            })
        };
        assert!(
            !callable.is_null(),
            "python operator source failed to compile/evaluate"
        );
        (
            PyOpState {
                interp,
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
        let callable = state.callable;
        let writes = state.writes_output;
        // Position of the first input in the args tuple (1 if `out` leads).
        let base = usize::from(writes);

        let ok = unsafe {
            with_interp(state.interp.0, || {
                let args = ffi::PyTuple_New((n + base) as ffi::Py_ssize_t);
                if args.is_null() {
                    ffi::PyErr_Print();
                    return false;
                }

                // arg 0 (write mode only): zero-copy writable view over the
                // operator's own (graph-invariant) output buffer.
                if writes {
                    let out_slice = output.as_mut_slice();
                    // `as_mut_ptr` (not `as_ptr`): the view is written through, so
                    // its pointer must carry write provenance (a `&mut` reborrow).
                    let mv = f64_output_view(out_slice.as_mut_ptr(), out_slice.len());
                    if mv.is_null() {
                        ffi::PyErr_Print();
                        ffi::Py_DECREF(args);
                        return false;
                    }
                    ffi::PyTuple_SetItem(args, 0, mv); // steals `mv`
                }

                // args[base..]: read-only views over per-input GC-owned copies.
                for i in 0..n {
                    let s = inputs.get(i).as_slice();
                    let mv = f64_input_view(s.as_ptr(), s.len());
                    if mv.is_null() {
                        ffi::PyErr_Print();
                        ffi::Py_DECREF(args); // also drops views already set
                        return false;
                    }
                    ffi::PyTuple_SetItem(args, (i + base) as ffi::Py_ssize_t, mv); // steals
                }

                let res = ffi::PyObject_CallObject(callable, args);
                let ok = if res.is_null() {
                    ffi::PyErr_Print();
                    false
                } else {
                    // Write mode: output already written in place; ignore `res`.
                    // Return mode: land the returned sequence into the output.
                    let landed = writes || pyseq_to_slice(res, output.as_mut_slice());
                    ffi::Py_DECREF(res);
                    landed
                };
                ffi::Py_DECREF(args);
                ok
            })
        };
        assert!(
            ok,
            "python operator raised, or returned a non-sequence / wrong-length result"
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
    use crate::flow::{Adapt, Const, Map};

    /// Return mode: a two-input element-wise Python sum. Inputs reach the
    /// callable as read-only f64 memoryviews; the returned list is landed.
    #[test]
    fn py_operator_return_mode() {
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

    /// Write mode: the callable writes results directly into the zero-copy
    /// output memoryview; its return value is ignored.
    #[test]
    fn py_operator_write_mode_zero_copy() {
        let mut b = GraphBuilder::new();
        let a = b.push(Adapt::new(Const(Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]))), ());
        let out = b.push(
            Adapt::new(PyOperator::writing(
                "lambda out, a: [out.__setitem__(i, a[i] * 2.0) for i in range(len(out))]",
                3,
            )),
            &[a][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(a) = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[2.0, 4.0, 6.0]);
    }

    /// Inputs are exposed as read-only f64 memoryviews: the operator reports the
    /// type/format/readonly flags it observed.
    #[test]
    fn py_operator_input_is_readonly_f64_memoryview() {
        let mut b = GraphBuilder::new();
        let a = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![1.0_f64, 2.0]))), ());
        let out = b.push(
            Adapt::new(PyOperator::writing(
                "lambda out, a: out.__setitem__(0, float(\
                   isinstance(a, memoryview) and a.format == 'd' and a.readonly and len(a) == 2))",
                1,
            )),
            &[a][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(a) = Array::from_vec(&[2], vec![1.0, 2.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[1.0]); // all checks passed
    }

    /// Safety regression for the audit's critical finding: an operator that
    /// stashes a CHILD slice of an input (`a[:]`) across generations, downstream
    /// of a `Map` that REALLOCATES its `Array<f64>` each tick. Because inputs are
    /// copied into GC-owned snapshots, the stashed child reads a live snapshot
    /// (the gen-1 value) on gen 2 — never freed Rust heap. With a zero-copy input
    /// view this would be a use-after-free of the reallocated upstream buffer.
    #[test]
    fn retained_input_child_view_is_snapshot_not_uaf() {
        let mut b = GraphBuilder::new();
        let src = b.push(Adapt::new(Const(Array::from_vec(&[1], vec![5.0_f64]))), ());
        // Allocating map: clones the input into a fresh Array (new Box) each tick.
        let mapped = b.push(
            Adapt::new(Map::new(|a: &Array<f64>| a.clone())),
            src,
        );
        // On the first call stash `a[:]` (a child view) into a module global; on
        // every call write out[0] from that stashed snapshot.
        let out = b.push(
            Adapt::new(PyOperator::writing(
                "lambda out, a: out.__setitem__(0, globals().setdefault('_S', a[:])[0])",
                1,
            )),
            &[mapped][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        // Gen 1: input 5.0 → stash snapshot [5.0] → out[0] = 5.0.
        *g.cell_mut(src) = Array::from_vec(&[1], vec![5.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[5.0]);

        // Gen 2: input 99.0; Map reallocates (old Box freed). The stashed child
        // still reads the gen-1 snapshot — safe — so out[0] stays 5.0.
        *g.cell_mut(src) = Array::from_vec(&[1], vec![99.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[5.0]);
    }

    /// True parallelism: K CPU-bound pure-Python operators, each in its own
    /// sub-interpreter, run far faster on a K-worker pool than on a serial one.
    #[test]
    fn py_operators_run_in_parallel() {
        const K: usize = 4;
        // Heavy pure-Python loop; writes [xs[0]] (the loop value is discarded
        // via * 0, so output == input — easy to assert). Write mode.
        let heavy = "lambda out, xs: out.__setitem__(0, \
                      sum((i * 1103515245 + 12345) % 2147483647 \
                      for i in range(400000)) * 0 + xs[0])";

        let mut b = GraphBuilder::new();
        let src = b.push(Adapt::new(Const(Array::scalar(0.0_f64))), ());
        let outs: Vec<_> = (0..K)
            .map(|_| b.push(Adapt::new(PyOperator::writing(heavy, 1)), &[src][..]))
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
