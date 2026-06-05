//! Class-based Python operator host (feature `pyflow`).
//!
//! Where [`PyOperator`](super::python::PyOperator) wraps a single lambda
//! (`f(*inputs) -> ndarray`), [`PyClassOperator`] hosts a full Python operator
//! object mirroring the legacy `tradingflow.operator.Operator` contract, so the
//! Python-resident operator layer (predictors / portfolios / traders / stateful
//! metrics) ports nearly verbatim:
//!
//! ```text
//! init(inputs, timestamp) -> state
//! compute(state, inputs, output, timestamp, produced) -> bool   # @staticmethod
//! ```
//!
//! `inputs` is a tuple of **views** ([`NativeArrayView`] for now), `output` is a
//! writable view, `timestamp` is TAI nanoseconds (from the driver [`Clock`]),
//! `produced` is a `tuple[bool, ...]` parallel to `inputs`, and `state` is a
//! Python object carried across ticks. The operator writes into `output` via
//! `output.write(ndarray)` and returns whether to notify downstream.
//!
//! The `source` is a Python *program* (statements) that, executed in the
//! operator's own globals (with `np`/`numpy` pre-injected), binds the operator
//! instance to the name `__op__`.
//!
//! # Data model
//!
//! Copy-based (like the legacy bridge): each input view's `value()` copies the
//! cell's `Array<f64>` (any shape) out to a fresh NumPy array, and `write()`
//! copies a NumPy array back into the output cell. Copies make retention safe
//! and the cost is negligible against the NumPy/SciPy math these operators run.
//!
//! # Scope
//!
//! Phase 1a: homogeneous `Array<f64>` inputs (`[Port<Array<f64>>]`). `SeriesView`
//! (history) and heterogeneous input tuples land in a follow-up so predictors /
//! portfolios that read recorded `Series` can be ported.
//!
//! # Safety
//!
//! [`NativeArrayView`] holds a raw pointer to a graph cell's `Array<f64>`, valid
//! only for the duration of one `compute` (the cell is borrowed by the engine
//! then). Views are created fresh each call and must not be retained past it
//! (operator code does not). The pointer for an *output* view carries write
//! provenance (`&mut`); input views are read-only (`value()` only). `unsafe
//! Send + Sync` on the view and state is sound because a node's `compute` runs
//! on one thread at a time and views never cross threads.

use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};

use flowgraph::typed::{Port, SliceNotify, SliceRefs};

use super::op::{Clock, Operator};
use crate::Array;

// ===========================================================================
// NativeArrayView — a Python-visible view over a cell's `Array<f64>`
// ===========================================================================

/// View over a graph cell's `Array<f64>`. `value()` copies out; `write()` copies
/// in (output views only). Valid only during the `compute`/`init` call that
/// created it.
#[pyclass]
pub struct NativeArrayView {
    ptr: *mut Array<f64>,
    writable: bool,
}

// SAFETY: see module docs — single-threaded per compute, never retained/shared.
unsafe impl Send for NativeArrayView {}
unsafe impl Sync for NativeArrayView {}

#[pymethods]
impl NativeArrayView {
    /// Copy the cell array into a fresh NumPy array of the cell's shape.
    fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        let arr = unsafe { &*self.ptr };
        let nd = ArrayD::from_shape_vec(IxDyn(arr.shape()), arr.as_slice().to_vec())
            .expect("array shape/len mismatch");
        PyArrayDyn::from_owned_array(py, nd)
    }

    /// Alias for [`value`](Self::value).
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.value(py)
    }

    /// NumPy array protocol, so `np.asarray(view)` / `np.log(view)` work.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = copy;
        let arr = self.value(py).into_any();
        match dtype {
            None => Ok(arr),
            Some(dt) => arr.call_method1("astype", (dt,)),
        }
    }

    /// Overwrite the output cell from a NumPy array of matching element count.
    fn write(&self, value: PyReadonlyArrayDyn<'_, f64>) -> PyResult<()> {
        if !self.writable {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot write to a read-only (input) view",
            ));
        }
        let arr = unsafe { &mut *self.ptr };
        let src = value.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("write: non-contiguous array: {e}"))
        })?;
        let dst = arr.as_mut_slice();
        if src.len() != dst.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "write: expected {} elements, got {}",
                dst.len(),
                src.len()
            )));
        }
        dst.copy_from_slice(src);
        Ok(())
    }

    /// Element shape of the cell array.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        unsafe { &*self.ptr }.shape().to_vec()
    }
}

impl NativeArrayView {
    /// Build a view; `writable` must be `false` unless `ptr` carries write
    /// provenance (an output `&mut`).
    fn bind_view<'py>(
        py: Python<'py>,
        ptr: *mut Array<f64>,
        writable: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(py, NativeArrayView { ptr, writable })?.into_any())
    }
}

// ===========================================================================
// PyClassOperator — hosts a Python operator object
// ===========================================================================

/// A class-based Python operator. `source` is a Python program that binds the
/// operator instance to `__op__`; the operator implements `init`/`compute` per
/// the legacy contract (see module docs).
pub struct PyClassOperator {
    source: String,
    out_shape: Vec<usize>,
    clock: Clock,
}

impl PyClassOperator {
    pub fn new(source: impl Into<String>, out_shape: Vec<usize>, clock: Clock) -> Self {
        Self {
            source: source.into(),
            out_shape,
            clock,
        }
    }
}

/// State: the operator instance, its Python state object, and the driver clock.
pub struct PyClassState {
    operator: Py<PyAny>,
    py_state: Py<PyAny>,
    clock: Clock,
}

/// Build a `tuple` of read-only [`NativeArrayView`]s over the inputs.
fn input_views<'py>(
    py: Python<'py>,
    inputs: &SliceRefs<'_, Port<Array<f64>>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let n = inputs.len();
    let mut views: Vec<Bound<'py, PyAny>> = Vec::with_capacity(n);
    for i in 0..n {
        let r: &Array<f64> = inputs.get(i);
        views.push(NativeArrayView::bind_view(py, r as *const Array<f64> as *mut Array<f64>, false)?);
    }
    PyTuple::new(py, views)
}

/// Build a `tuple[bool, ...]` parallel to the inputs from the notify tree.
fn produced_tuple<'py>(
    py: Python<'py>,
    produced: &SliceNotify<'_, Port<Array<f64>>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let bits: Vec<bool> = (0..produced.len()).map(|i| produced.get(i)).collect();
    PyTuple::new(py, bits)
}

impl Operator for PyClassOperator {
    type Inputs = [Port<Array<f64>>];
    type Output = Array<f64>;
    type State = PyClassState;

    fn init(&self, inputs: SliceRefs<'_, Port<Array<f64>>>) -> (PyClassState, Array<f64>) {
        let code = CString::new(self.source.as_str()).expect("python source has interior NUL");
        let ts = self.clock.get().as_nanos();
        let (operator, py_state) = Python::attach(|py| {
            let run = || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                let globals = PyDict::new(py);
                let np = py.import("numpy")?;
                globals.set_item("np", &np)?;
                globals.set_item("numpy", &np)?;
                py.run(code.as_c_str(), Some(&globals), None)?;
                let operator = globals
                    .get_item("__op__")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "python operator source must bind `__op__`",
                        )
                    })?;
                let views = input_views(py, &inputs)?;
                let state = operator.call_method1("init", (views, ts))?;
                Ok((operator.unbind(), state.unbind()))
            };
            run().unwrap_or_else(|e| {
                e.print(py);
                panic!("python operator init failed");
            })
        });
        (
            PyClassState {
                operator,
                py_state,
                clock: self.clock.clone(),
            },
            Array::zeros(&self.out_shape),
        )
    }

    fn compute(
        state: &mut PyClassState,
        inputs: SliceRefs<'_, Port<Array<f64>>>,
        output: &mut Array<f64>,
        produced: SliceNotify<'_, Port<Array<f64>>>,
    ) -> bool {
        let ts = state.clock.get().as_nanos();
        let out_ptr = output as *mut Array<f64>;
        // Distinguish a legitimate `False` (no notify) from a Python error.
        let result: Result<bool, ()> = Python::attach(|py| {
            let run = || -> PyResult<bool> {
                let views = input_views(py, &inputs)?;
                let out_view = NativeArrayView::bind_view(py, out_ptr, true)?;
                let prod = produced_tuple(py, &produced)?;
                let operator = state.operator.bind(py);
                let result = operator.call_method1(
                    "compute",
                    (state.py_state.bind(py), views, out_view, ts, prod),
                )?;
                result.extract::<bool>()
            };
            run().map_err(|e| e.print(py))
        });
        result.unwrap_or_else(|()| panic!("python operator compute failed (see traceback above)"))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::PyClassOperator;
    use flowgraph::core::Pool;
    use flowgraph::typed::{Graph, GraphBuilder};

    use crate::Array;
    use crate::flow::{Adapt, Clock, Const};

    /// Turnover ported ~verbatim from python/.../metrics/turnover.py: a stateful
    /// operator (caches prev weights, warmup returns False) over one Array input.
    /// Raw string at column 0 so Python indentation is preserved literally.
    const TURNOVER: &str = r#"
import numpy as np
from dataclasses import dataclass
@dataclass
class S:
    prev: object = None
    initialized: bool = False
class Turnover:
    def init(self, inputs, timestamp):
        return S()
    @staticmethod
    def compute(state, inputs, output, timestamp, produced):
        current = np.where(np.isfinite(inputs[0].value()), inputs[0].value(), 0.0)
        if not state.initialized:
            state.prev = current
            state.initialized = True
            return False
        turnover = float(np.sum(np.abs(current - state.prev)))
        state.prev = current
        output.write(np.array(turnover, dtype=np.float64))
        return True
__op__ = Turnover()
"#;

    #[test]
    fn py_class_operator_turnover() {
        let clock = Clock::new();
        let mut b = GraphBuilder::new();
        let src = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![0.5_f64, 0.5]))), ());
        let out = b.push(
            Adapt::new(PyClassOperator::new(TURNOVER, vec![], clock.clone())),
            &[src][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        // Tick 1: warmup — caches [0.5, 0.5], returns False, output stays 0.
        *g.cell_mut(src) = Array::from_vec(&[2], vec![0.5, 0.5]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[0.0]);

        // Tick 2: turnover = |0.3-0.5| + |0.7-0.5| = 0.4.
        *g.cell_mut(src) = Array::from_vec(&[2], vec![0.3, 0.7]);
        g.stabilize(&mut pool);
        assert!((g.cell(out).as_slice()[0] - 0.4).abs() < 1e-12);

        // Tick 3: turnover = |1.0-0.3| + |0.0-0.7| = 1.4.
        *g.cell_mut(src) = Array::from_vec(&[2], vec![1.0, 0.0]);
        g.stabilize(&mut pool);
        assert!((g.cell(out).as_slice()[0] - 1.4).abs() < 1e-12);
    }
}
