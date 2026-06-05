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
//! `inputs` is a tuple of **views** — [`NativeArrayView`] for `Array<f64>` cells
//! and [`NativeSeriesView`] for `Series<f64>` cells (history), `None` for unit
//! (clock) inputs. `output` is a writable array view, `timestamp` is TAI
//! nanoseconds (from the driver [`Clock`]), `produced` is a `tuple[bool, ...]`
//! parallel to `inputs`, and `state` is a Python object carried across ticks.
//!
//! The `source` is a Python *program* (statements) executed in the operator's
//! own globals (with `np`/`numpy` pre-injected) that binds the operator instance
//! to the name `__op__`.
//!
//! # Heterogeneous inputs
//!
//! flowgraph's typed ports are homogeneous slices or fixed tuples, so an
//! operator's input shape is its concrete [`Ports`] type, e.g.
//! `(Port<Array<f64>>, Port<Series<f64>>, Port<Series<f64>>)` for a predictor or
//! `[Port<Array<f64>>]` for an all-array operator. The [`PyArgs`] trait walks
//! that type to build the view tuple + produced bools. (An erased enum input
//! would have to clone growing `Series` each tick — `PyArgs` reads the borrowed
//! cells directly instead.)
//!
//! # Data model
//!
//! Copy-based (like the legacy bridge): each view copies the cell data out to a
//! fresh NumPy array on read, and `output.write()` copies a NumPy array back.
//! Copies make retention safe and the cost is negligible against the NumPy/SciPy
//! math these operators run.
//!
//! # Safety
//!
//! The view pyclasses hold a raw pointer to a graph cell, valid only for the
//! duration of one `compute`/`init` (the cell is borrowed by the engine then).
//! Views are created fresh each call and must not be retained past it. The
//! output array view's pointer carries write provenance (`&mut`); input views
//! are read-only. `unsafe Send + Sync` is sound: a node's `compute` runs on one
//! thread at a time and views never cross threads.

use std::ffi::CString;
use std::marker::PhantomData;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArrayDyn};

use flowgraph::typed::{Port, Ports, SliceNotify, SliceRefs};

use super::op::{Clock, Operator};
use crate::{Array, Series};

// ===========================================================================
// NativeArrayView — Python-visible view over a cell's `Array<f64>`
// ===========================================================================

/// View over a graph cell's `Array<f64>`. `value()` copies out; `write()` copies
/// in (output views only). Valid only during the call that created it.
#[pyclass]
pub struct NativeArrayView {
    ptr: *mut Array<f64>,
    writable: bool,
}

// SAFETY: single-threaded per compute, never retained/shared (module docs).
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
            return Err(PyValueError::new_err("cannot write to a read-only (input) view"));
        }
        let arr = unsafe { &mut *self.ptr };
        let src = value
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("write: non-contiguous array: {e}")))?;
        let dst = arr.as_mut_slice();
        if src.len() != dst.len() {
            return Err(PyValueError::new_err(format!(
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
    fn bind<'py>(py: Python<'py>, ptr: *mut Array<f64>, writable: bool) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(py, NativeArrayView { ptr, writable })?.into_any())
    }
}

// ===========================================================================
// NativeSeriesView — Python-visible view over a cell's `Series<f64>` (history)
// ===========================================================================

/// Read-only view over a graph cell's `Series<f64>`: positional history access
/// matching the legacy `SeriesView`. Valid only during the call that created it.
#[pyclass]
pub struct NativeSeriesView {
    ptr: *const Series<f64>,
}

// SAFETY: single-threaded per compute, never retained/shared (module docs).
unsafe impl Send for NativeSeriesView {}
unsafe impl Sync for NativeSeriesView {}

#[pymethods]
impl NativeSeriesView {
    fn __len__(&self) -> usize {
        unsafe { &*self.ptr }.len()
    }

    /// Element shape (without the time axis).
    #[getter]
    fn shape(&self) -> Vec<usize> {
        unsafe { &*self.ptr }.shape().to_vec()
    }

    /// Values in `[start, end)` as a `(end-start, *element_shape)` NumPy array.
    #[pyo3(signature = (start=0, end=None))]
    fn values<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: Option<usize>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let s = unsafe { &*self.ptr };
        let n = s.len();
        let start = start.min(n);
        let end = end.unwrap_or(n).min(n).max(start);
        let stride = s.stride();
        let flat = &s.values()[start * stride..end * stride];
        let mut full = vec![end - start];
        full.extend_from_slice(s.shape());
        let nd = ArrayD::from_shape_vec(IxDyn(&full), flat.to_vec()).expect("series shape mismatch");
        PyArrayDyn::from_owned_array(py, nd)
    }

    /// Most recent element as an `element_shape` NumPy array.
    fn last<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let s = unsafe { &*self.ptr };
        let last = s.last().ok_or_else(|| PyIndexError::new_err("last() on empty series"))?;
        let nd = ArrayD::from_shape_vec(IxDyn(s.shape()), last.to_vec()).expect("series shape mismatch");
        Ok(PyArrayDyn::from_owned_array(py, nd))
    }

    /// Element at positional index `i` (supports negative indexing).
    fn at<'py>(&self, py: Python<'py>, i: isize) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let s = unsafe { &*self.ptr };
        let n = s.len() as isize;
        let idx = if i < 0 { n + i } else { i };
        if idx < 0 || idx >= n {
            return Err(PyIndexError::new_err(format!("index {i} out of bounds (len {n})")));
        }
        let elem = s.at(idx as usize);
        let nd = ArrayD::from_shape_vec(IxDyn(s.shape()), elem.to_vec()).expect("series shape mismatch");
        Ok(PyArrayDyn::from_owned_array(py, nd))
    }

    /// Timestamps in `[start, end)` as an int64 (TAI ns) NumPy array.
    #[pyo3(signature = (start=0, end=None))]
    fn slice<'py>(&self, py: Python<'py>, start: usize, end: Option<usize>) -> Bound<'py, PyArray1<i64>> {
        let s = unsafe { &*self.ptr };
        let n = s.len();
        let start = start.min(n);
        let end = end.unwrap_or(n).min(n).max(start);
        let ts: Vec<i64> = s.timestamps()[start..end].iter().map(|t| t.as_nanos()).collect();
        PyArray1::from_slice(py, &ts)
    }

    /// Positional indexing: `int` -> single element, contiguous `slice` -> range.
    fn __getitem__<'py>(&self, py: Python<'py>, key: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            return Ok(self.at(py, i)?.into_any());
        }
        let sl = key.cast::<PySlice>().map_err(|_| {
            PyValueError::new_err("series index must be an int or a contiguous slice")
        })?;
        let n = unsafe { &*self.ptr }.len();
        let ind = sl.indices(n as isize)?;
        if ind.step != 1 {
            return Err(PyValueError::new_err("only contiguous (step 1) slices supported"));
        }
        Ok(self.values(py, ind.start as usize, Some(ind.stop as usize)).into_any())
    }
}

impl NativeSeriesView {
    fn bind<'py>(py: Python<'py>, ptr: *const Series<f64>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(py, NativeSeriesView { ptr })?.into_any())
    }
}

// ===========================================================================
// PyArgs — build the Python view tuple + produced bools from typed input refs
// ===========================================================================

/// Walks an operator's [`Ports`] input type, appending one Python view per leaf
/// ([`NativeArrayView`] / [`NativeSeriesView`] / `None`) and one bool per leaf.
pub trait PyArgs: Ports {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Refs<'_>,
        out: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()>;

    fn append_produced(notify: Self::Notify<'_>, out: &mut Vec<bool>);
}

impl PyArgs for Port<Array<f64>> {
    fn append_views<'py>(py: Python<'py>, refs: &Array<f64>, out: &mut Vec<Bound<'py, PyAny>>) -> PyResult<()> {
        out.push(NativeArrayView::bind(py, refs as *const Array<f64> as *mut Array<f64>, false)?);
        Ok(())
    }
    fn append_produced(notify: bool, out: &mut Vec<bool>) {
        out.push(notify);
    }
}

impl PyArgs for Port<Series<f64>> {
    fn append_views<'py>(py: Python<'py>, refs: &Series<f64>, out: &mut Vec<Bound<'py, PyAny>>) -> PyResult<()> {
        out.push(NativeSeriesView::bind(py, refs as *const Series<f64>)?);
        Ok(())
    }
    fn append_produced(notify: bool, out: &mut Vec<bool>) {
        out.push(notify);
    }
}

impl PyArgs for Port<()> {
    fn append_views<'py>(py: Python<'py>, _refs: &(), out: &mut Vec<Bound<'py, PyAny>>) -> PyResult<()> {
        out.push(py.None().into_bound(py));
        Ok(())
    }
    fn append_produced(notify: bool, out: &mut Vec<bool>) {
        out.push(notify);
    }
}

impl<T: PyArgs + 'static> PyArgs for [T] {
    fn append_views<'py>(py: Python<'py>, refs: SliceRefs<'_, T>, out: &mut Vec<Bound<'py, PyAny>>) -> PyResult<()> {
        for i in 0..refs.len() {
            T::append_views(py, refs.get(i), out)?;
        }
        Ok(())
    }
    fn append_produced(notify: SliceNotify<'_, T>, out: &mut Vec<bool>) {
        for i in 0..notify.len() {
            T::append_produced(notify.get(i), out);
        }
    }
}

macro_rules! tuple_pyargs {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: PyArgs,)+> PyArgs for ($($T,)+) {
            fn append_views<'py>(
                py: Python<'py>,
                refs: Self::Refs<'_>,
                out: &mut Vec<Bound<'py, PyAny>>,
            ) -> PyResult<()> {
                $( $T::append_views(py, refs.$idx, out)?; )+
                Ok(())
            }
            fn append_produced(notify: Self::Notify<'_>, out: &mut Vec<bool>) {
                $( $T::append_produced(notify.$idx, out); )+
            }
        }
    };
}

tuple_pyargs!(0: A, 1: B);
tuple_pyargs!(0: A, 1: B, 2: C);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);

// ===========================================================================
// PyClassOperator — hosts a Python operator object over inputs `I`
// ===========================================================================

/// A class-based Python operator over input ports `I` (e.g. `[Port<Array<f64>>]`
/// or `(Port<Array<f64>>, Port<Series<f64>>, Port<Series<f64>>)`). `source` is a
/// Python program binding the operator instance to `__op__`.
pub struct PyClassOperator<I: PyArgs + ?Sized = [Port<Array<f64>>]> {
    source: String,
    out_shape: Vec<usize>,
    clock: Clock,
    _marker: PhantomData<fn() -> Box<I>>,
}

impl<I: PyArgs + ?Sized> PyClassOperator<I> {
    pub fn new(source: impl Into<String>, out_shape: Vec<usize>, clock: Clock) -> Self {
        Self {
            source: source.into(),
            out_shape,
            clock,
            _marker: PhantomData,
        }
    }
}

/// State: the operator instance, its Python state object, and the driver clock.
pub struct PyClassState {
    operator: Py<PyAny>,
    py_state: Py<PyAny>,
    clock: Clock,
}

impl<I: PyArgs + ?Sized + 'static> Operator for PyClassOperator<I> {
    type Inputs = I;
    type Output = Array<f64>;
    type State = PyClassState;

    fn init(&self, inputs: I::Refs<'_>) -> (PyClassState, Array<f64>) {
        let code = CString::new(self.source.as_str()).expect("python source has interior NUL");
        let ts = self.clock.get().as_nanos();
        let (operator, py_state) = Python::attach(|py| {
            let run = || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                let globals = PyDict::new(py);
                let np = py.import("numpy")?;
                globals.set_item("np", &np)?;
                globals.set_item("numpy", &np)?;
                py.run(code.as_c_str(), Some(&globals), None)?;
                let operator = globals.get_item("__op__")?.ok_or_else(|| {
                    PyValueError::new_err("python operator source must bind `__op__`")
                })?;
                let mut views: Vec<Bound<'_, PyAny>> = Vec::new();
                I::append_views(py, inputs, &mut views)?;
                let state = operator.call_method1("init", (PyTuple::new(py, views)?, ts))?;
                Ok((operator.unbind(), state.unbind()))
            };
            run().unwrap_or_else(|e| {
                e.print(py);
                panic!("python operator init failed (see traceback above)");
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
        inputs: I::Refs<'_>,
        output: &mut Array<f64>,
        produced: I::Notify<'_>,
    ) -> bool {
        let ts = state.clock.get().as_nanos();
        let out_ptr = output as *mut Array<f64>;
        let result: Result<bool, ()> = Python::attach(|py| {
            let run = || -> PyResult<bool> {
                let mut views: Vec<Bound<'_, PyAny>> = Vec::new();
                I::append_views(py, inputs, &mut views)?;
                let mut bits: Vec<bool> = Vec::new();
                I::append_produced(produced, &mut bits);
                let out_view = NativeArrayView::bind(py, out_ptr, true)?;
                let operator = state.operator.bind(py);
                operator
                    .call_method1(
                        "compute",
                        (
                            state.py_state.bind(py),
                            PyTuple::new(py, views)?,
                            out_view,
                            ts,
                            PyTuple::new(py, bits)?,
                        ),
                    )?
                    .extract::<bool>()
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
    use flowgraph::typed::{Graph, GraphBuilder, Port};

    use crate::Instant;
    use crate::flow::{Adapt, Clock, Const, Record};
    use crate::{Array, Series};

    /// Turnover ported ~verbatim from python/.../metrics/turnover.py: a stateful
    /// operator over one Array input. Raw string at column 0 preserves Python
    /// indentation (Rust `\`-continuation would strip it).
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
            Adapt::new(PyClassOperator::<[Port<Array<f64>>]>::new(TURNOVER, vec![], clock.clone())),
            &[src][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(src) = Array::from_vec(&[2], vec![0.5, 0.5]);
        g.stabilize(&mut pool);
        assert_eq!(g.cell(out).as_slice(), &[0.0]); // warmup

        *g.cell_mut(src) = Array::from_vec(&[2], vec![0.3, 0.7]);
        g.stabilize(&mut pool);
        assert!((g.cell(out).as_slice()[0] - 0.4).abs() < 1e-12);

        *g.cell_mut(src) = Array::from_vec(&[2], vec![1.0, 0.0]);
        g.stabilize(&mut pool);
        assert!((g.cell(out).as_slice()[0] - 1.4).abs() < 1e-12);
    }

    /// Heterogeneous inputs: an (Array, Series) operator that reads Series
    /// history. Proves NativeSeriesView (values/len/getitem) + tuple PyArgs.
    /// Computes: output = mean over history of (series[-1] dotted with weights).
    const HIST_DOT: &str = r#"
import numpy as np
class HistDot:
    def init(self, inputs, timestamp):
        return {}
    @staticmethod
    def compute(state, inputs, output, timestamp, produced):
        weights = inputs[0].value()          # (N,)
        hist = inputs[1].values()            # (T, N)
        # mean over time of <hist[t], weights>
        val = float(np.mean(hist @ weights)) if len(inputs[1]) > 0 else 0.0
        output.write(np.array(val, dtype=np.float64))
        return True
__op__ = HistDot()
"#;

    #[test]
    fn py_class_operator_heterogeneous_series() {
        let clock = Clock::new();
        let mut b = GraphBuilder::new();
        // weights: Array(2); feed_data: Array(2) recorded into a Series(2).
        let weights = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![1.0_f64, 1.0]))), ());
        let feed = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![0.0_f64, 0.0]))), ());
        // Record needs the clock; build via Record::new(clock).
        let series = b.push(Adapt::new(Record::new(clock.clone())), feed);
        let out = b.push(
            Adapt::new(PyClassOperator::<(Port<Array<f64>>, Port<Series<f64>>)>::new(
                HIST_DOT,
                vec![],
                clock.clone(),
            )),
            (weights, series),
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        // Tick 1 @ t=100: feed [1,2]; series=[[1,2]]; dot with [1,1]=3; mean=3.
        clock.set(Instant::from_nanos(100));
        *g.cell_mut(weights) = Array::from_vec(&[2], vec![1.0, 1.0]);
        *g.cell_mut(feed) = Array::from_vec(&[2], vec![1.0, 2.0]);
        g.stabilize(&mut pool);
        assert!((g.cell(out).as_slice()[0] - 3.0).abs() < 1e-12);

        // Tick 2 @ t=200: feed [3,4]; series=[[1,2],[3,4]]; dots=3,7; mean=5.
        clock.set(Instant::from_nanos(200));
        *g.cell_mut(feed) = Array::from_vec(&[2], vec![3.0, 4.0]);
        g.stabilize(&mut pool);
        assert!((g.cell(out).as_slice()[0] - 5.0).abs() < 1e-12);
    }
}
