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

/// A typed keyword argument passed to a Python operator's `build(**kwargs)`.
#[derive(Clone)]
enum Param {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

/// Keyword arguments for a Python operator factory. Build with the chainable
/// setters, e.g. `PyParams::new().int("num_stocks", 500).float("lam", 0.1)`.
#[derive(Clone, Default)]
pub struct PyParams(Vec<(String, Param)>);

impl PyParams {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn int(mut self, k: &str, v: i64) -> Self {
        self.0.push((k.into(), Param::Int(v)));
        self
    }
    pub fn float(mut self, k: &str, v: f64) -> Self {
        self.0.push((k.into(), Param::Float(v)));
        self
    }
    pub fn bool(mut self, k: &str, v: bool) -> Self {
        self.0.push((k.into(), Param::Bool(v)));
        self
    }
    pub fn str(mut self, k: &str, v: impl Into<String>) -> Self {
        self.0.push((k.into(), Param::Str(v.into())));
        self
    }
    pub fn ints(mut self, k: &str, v: Vec<i64>) -> Self {
        self.0.push((k.into(), Param::Ints(v)));
        self
    }
    pub fn floats(mut self, k: &str, v: Vec<f64>) -> Self {
        self.0.push((k.into(), Param::Floats(v)));
        self
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.0 {
            match v {
                Param::Int(x) => d.set_item(k, x)?,
                Param::Float(x) => d.set_item(k, x)?,
                Param::Bool(x) => d.set_item(k, x)?,
                Param::Str(x) => d.set_item(k, x)?,
                Param::Ints(x) => d.set_item(k, x.clone())?,
                Param::Floats(x) => d.set_item(k, x.clone())?,
            }
        }
        Ok(d)
    }
}

/// Where a Python operator's definition comes from. Each defines a factory
/// `build(**kwargs) -> operator` (called with [`PyParams`]) or binds the
/// instance to `__op__` directly.
#[derive(Clone)]
enum Loader {
    /// Inline Python program (statements).
    Source(String),
    /// An importable module on the embedded interpreter's path.
    Module(String),
}

/// Resolve the operator instance from its [`Loader`] + [`PyParams`].
fn resolve_operator<'py>(
    py: Python<'py>,
    loader: &Loader,
    params: &PyParams,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = params.to_dict(py)?;
    let (build, op_obj) = match loader {
        Loader::Source(src) => {
            let g = PyDict::new(py);
            let np = py.import("numpy")?;
            g.set_item("np", &np)?;
            g.set_item("numpy", &np)?;
            let code = CString::new(src.as_str())
                .map_err(|_| PyValueError::new_err("python source has interior NUL"))?;
            py.run(code.as_c_str(), Some(&g), None)?;
            (g.get_item("build")?, g.get_item("__op__")?)
        }
        Loader::Module(name) => {
            let module = py.import(name.as_str())?;
            (module.getattr("build").ok(), module.getattr("__op__").ok())
        }
    };
    if let Some(build) = build {
        build.call((), Some(&kwargs))
    } else if let Some(op) = op_obj {
        Ok(op)
    } else {
        Err(PyValueError::new_err(
            "python operator must define `build(**kwargs)` or bind `__op__`",
        ))
    }
}

/// A class-based Python operator over input ports `I` (e.g. `[Port<Array<f64>>]`
/// or `(Port<Array<f64>>, Port<Series<f64>>, Port<Series<f64>>)`). Load its
/// definition from a `.py` file, an importable module, or an inline source; each
/// defines `build(**kwargs)` (called with [`PyParams`]) or binds `__op__`.
pub struct PyClassOperator<I: PyArgs + ?Sized = [Port<Array<f64>>]> {
    loader: Loader,
    params: PyParams,
    out_shape: Vec<usize>,
    clock: Clock,
    _marker: PhantomData<fn() -> Box<I>>,
}

impl<I: PyArgs + ?Sized> PyClassOperator<I> {
    /// Load from a `.py` file on disk (read now).
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
        params: PyParams,
        out_shape: Vec<usize>,
        clock: Clock,
    ) -> Self {
        let path = path.as_ref();
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("cannot read python operator file {}: {e}", path.display())
        });
        Self::from_source(src, params, out_shape, clock)
    }

    /// Load from an importable module (on the embedded interpreter's path).
    pub fn from_module(
        module: impl Into<String>,
        params: PyParams,
        out_shape: Vec<usize>,
        clock: Clock,
    ) -> Self {
        Self {
            loader: Loader::Module(module.into()),
            params,
            out_shape,
            clock,
            _marker: PhantomData,
        }
    }

    /// Load from an inline Python program (handy for tests / one-offs).
    pub fn from_source(
        source: impl Into<String>,
        params: PyParams,
        out_shape: Vec<usize>,
        clock: Clock,
    ) -> Self {
        Self {
            loader: Loader::Source(source.into()),
            params,
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
        let ts = self.clock.get().as_nanos();
        let (operator, py_state) = Python::attach(|py| {
            let run = || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                let operator = resolve_operator(py, &self.loader, &self.params)?;
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
    use super::{PyClassOperator, PyParams};
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
            Adapt::new(PyClassOperator::<[Port<Array<f64>>]>::from_source(
                TURNOVER,
                PyParams::new(),
                vec![],
                clock.clone(),
            )),
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
            Adapt::new(PyClassOperator::<(Port<Array<f64>>, Port<Series<f64>>)>::from_source(
                HIST_DOT,
                PyParams::new(),
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

    /// Loading an operator from a plain `.py` file via a `build(**kwargs)`
    /// factory parameterized from Rust with [`PyParams`].
    #[test]
    fn py_class_operator_from_file_with_params() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scaler.py");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(
                br#"
import numpy as np
class Scaler:
    def __init__(self, scale):
        self.scale = scale
    def init(self, inputs, timestamp):
        return {"scale": self.scale}
    @staticmethod
    def compute(state, inputs, output, timestamp, produced):
        total = float(np.sum(inputs[0].value())) * state["scale"]
        output.write(np.array(total, dtype=np.float64))
        return True
def build(scale=1.0):
    return Scaler(scale)
"#,
            )
            .unwrap();

        let clock = Clock::new();
        let mut b = GraphBuilder::new();
        let src = b.push(Adapt::new(Const(Array::from_vec(&[4], vec![1.0_f64, 2.0, 3.0, 4.0]))), ());
        let out = b.push(
            Adapt::new(PyClassOperator::<[Port<Array<f64>>]>::from_file(
                &path,
                PyParams::new().float("scale", 3.0),
                vec![],
                clock.clone(),
            )),
            &[src][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(src) = Array::from_vec(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        g.stabilize(&mut pool);
        assert!((g.cell(out).as_slice()[0] - 30.0).abs() < 1e-12); // sum(1..4)=10 * 3
    }

    // -- Integration with the real `flowops` package (needs python/ on -----------
    //    PYTHONPATH + numpy on the embedded interpreter's path). -----------------

    /// The ported Turnover operator loaded from `flowops.metrics.turnover` runs
    /// through the real Rust host + engine (not just the pure-Python testkit).
    #[test]
    fn flowops_turnover_via_module() {
        let clock = Clock::new();
        let mut b = GraphBuilder::new();
        let src = b.push(Adapt::new(Const(Array::from_vec(&[2], vec![0.5_f64, 0.5]))), ());
        let out = b.push(
            Adapt::new(PyClassOperator::<[Port<Array<f64>>]>::from_module(
                "flowops.metrics.turnover",
                PyParams::new().int("num_stocks", 2),
                vec![],
                clock.clone(),
            )),
            &[src][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        *g.cell_mut(src) = Array::from_vec(&[2], vec![0.5, 0.5]);
        g.stabilize(&mut pool); // warmup
        *g.cell_mut(src) = Array::from_vec(&[2], vec![0.3, 0.7]);
        g.stabilize(&mut pool);
        assert!((g.cell(out).as_slice()[0] - 0.4).abs() < 1e-12);
    }

    /// A real predictor (LinearRegression) end-to-end: Array universe + two
    /// Series-history inputs (features (N,F), target (N)), produced-gated
    /// rebalance, emitting an (N,) prediction. Validates the heterogeneous
    /// Series path through the engine.
    #[test]
    fn flowops_linear_regression_predictor() {
        const N: usize = 3;
        const F: usize = 2;
        let clock = Clock::new();
        let mut b = GraphBuilder::new();
        let universe = b.push(Adapt::new(Const(Array::from_vec(&[N], vec![1.0; N]))), ());
        let feat_feed = b.push(Adapt::new(Const(Array::zeros(&[N, F]))), ());
        let tgt_feed = b.push(Adapt::new(Const(Array::zeros(&[N]))), ());
        let feat_series = b.push(Adapt::new(Record::new(clock.clone())), feat_feed);
        let tgt_series = b.push(Adapt::new(Record::new(clock.clone())), tgt_feed);
        let pred = b.push(
            Adapt::new(
                PyClassOperator::<(Port<Array<f64>>, Port<Series<f64>>, Port<Series<f64>>)>::from_module(
                    "flowops.predictors.mean.linear_regression",
                    PyParams::new()
                        .int("num_stocks", N as i64)
                        .int("num_features", F as i64)
                        .int("universe_size", N as i64)
                        .int("target_offset", 1),
                    vec![N],
                    clock.clone(),
                ),
            ),
            (universe, feat_series, tgt_series),
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        // Feed a few ticks of features/targets with a linear relationship so the
        // pooled OLS is well-posed; rebalance each tick (universe produces).
        for t in 1..=5_i64 {
            let x: Vec<f64> = (0..N * F).map(|k| (t as f64) + 0.1 * k as f64).collect();
            let y: Vec<f64> = (0..N).map(|i| 0.5 * (t as f64) + i as f64).collect();
            clock.set(Instant::from_nanos(t * 100));
            *g.cell_mut(feat_feed) = Array::from_vec(&[N, F], x);
            *g.cell_mut(tgt_feed) = Array::from_vec(&[N], y);
            *g.cell_mut(universe) = Array::from_vec(&[N], vec![1.0; N]);
            g.stabilize(&mut pool);
        }

        let mu = g.cell(pred).as_slice();
        assert_eq!(mu.len(), N);
        assert!(mu.iter().all(|v| v.is_finite()), "prediction has non-finite entries: {mu:?}");
    }
}
