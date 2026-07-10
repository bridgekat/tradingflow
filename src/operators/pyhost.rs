//! Class-based Python operator host (feature `python`).
//!
//! Where [`PyOperator`](super::python::PyOperator) wraps a single lambda
//! (`f(*inputs) -> ndarray`), [`PyClassOperator`] hosts a full Python operator
//! object mirroring the legacy `tradingflow.operator.Operator` contract, so the
//! Python-resident operator layer (predictors / portfolios / stateful
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
//! flowgraph's typed interfaces are trees of `RefPort<T>` leaves, runtime-length
//! `RefPorts<T>` groups, and tuples, so an operator's input shape is its concrete
//! [`Interface`] type, e.g.
//! `(RefPort<Array<f64>>, RefPort<Series<f64>>, RefPort<Series<f64>>)` for a predictor or
//! `RefPorts<Array<f64>>` for an all-array operator. The [`PyArgs`] trait walks
//! that type's refs to build the view tuple + produced bools in one pass. (An
//! erased enum input would have to clone growing `Series` each tick — `PyArgs`
//! reads the borrowed cells directly instead.)
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

use flowgraph::typed::{Interface, InterfaceHandles, Operator, RefPort, RefPorts};

use super::op::Clock;
use crate::{Array, Instant, Series};

// ===========================================================================
// NativeArrayView — Python-visible view over a cell's `Array<f64, N>`
// ===========================================================================
//
// [array-view-refactor] The const-rank refactor replaces the dynamic-rank
// `*mut Array<f64>` with a rank-erased pointer + an inline extents vector.
// Each per-leaf `PyArgs` impl knows its concrete rank `N` at the call site and
// reads the cell's static `[usize; N]` extents there; the pyclass stores them
// as a `Vec<usize>` so a single non-generic pyclass serves every rank (PyO3
// `#[pyclass]` cannot itself be const-generic). The flat `f64` buffer and a
// raw byte length are enough for the copy-out / copy-in, since `Array<f64, N>`
// is row-major contiguous — `as_slice()`/`as_mut_slice()` over the whole buffer
// are rank-independent. The output view reconstructs the typed `&mut Array<f64,
// NO>` from the stored pointer + rank tag (see [`NativeArrayView::bind`]).

/// View over a graph cell's `Array<f64, N>`. `value()` copies out; `write()`
/// copies in (output views only). Valid only during the call that created it.
#[pyclass]
pub struct NativeArrayView {
    /// Pointer to the flat row-major `f64` buffer (`Array<f64, N>::as_slice`).
    data: *mut f64,
    /// Number of scalars (product of extents).
    len: usize,
    /// Element extents (the cell's static `[usize; N]`, as a runtime vector).
    extents: Vec<usize>,
    writable: bool,
}

// SAFETY: single-threaded per compute, never retained/shared (module docs).
unsafe impl Send for NativeArrayView {}
unsafe impl Sync for NativeArrayView {}

#[pymethods]
impl NativeArrayView {
    /// Copy the cell array into a fresh NumPy array of the cell's shape.
    fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        let src = unsafe { std::slice::from_raw_parts(self.data, self.len) };
        let nd = ArrayD::from_shape_vec(IxDyn(&self.extents), src.to_vec())
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
        let src = value
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("write: non-contiguous array: {e}")))?;
        if src.len() != self.len {
            return Err(PyValueError::new_err(format!(
                "write: expected {} elements, got {}",
                self.len,
                src.len()
            )));
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(self.data, self.len) };
        dst.copy_from_slice(src);
        Ok(())
    }

    /// Element shape of the cell array.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.extents.clone()
    }
}

impl NativeArrayView {
    /// Bind a view over an `Array<f64, N>` (input: read-only, output: writable).
    /// The concrete rank `N` is known at the call site, so the static extents
    /// are read here and stored runtime-wide; the flat buffer is rank-agnostic.
    fn bind<'py, const N: usize>(
        py: Python<'py>,
        arr: *mut Array<f64, N>,
        writable: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // SAFETY: `arr` is a live cell for the duration of this call (module docs).
        let r = unsafe { &mut *arr };
        let view = NativeArrayView {
            data: r.as_mut_slice().as_mut_ptr(),
            len: r.len(),
            extents: r.extents().to_vec(),
            writable,
        };
        Ok(Bound::new(py, view)?.into_any())
    }
}

// ===========================================================================
// NativeSeriesView — Python-visible view over a cell's `Series<f64>` (history)
// ===========================================================================

/// Read-only view over a graph cell's `Series<f64, N>`: positional history
/// access (logical indices) matching the legacy Python `SeriesView`. Valid
/// only during the call that created it. Rank-erased like
/// [`NativeArrayView`]: the per-leaf [`PyArgs`] impl knows the concrete rank
/// `N` at the bind site and captures the retained window's raw parts plus the
/// logical bookkeeping there, so a single non-generic pyclass serves every
/// rank.
#[pyclass]
pub struct NativeSeriesView {
    /// Retained window: flat row-major values (`retained * stride` scalars).
    values: *const f64,
    /// Retained window: one timestamp per element.
    timestamps: *const Instant,
    /// Number of physically retained elements.
    retained: usize,
    /// Logical index of retained element 0 (count of evicted elements).
    base: usize,
    /// Scalars per element.
    stride: usize,
    /// Element extents (the cell's static `[usize; N]`, as a runtime vector).
    extents: Vec<usize>,
}

// SAFETY: single-threaded per compute, never retained/shared (module docs).
unsafe impl Send for NativeSeriesView {}
unsafe impl Sync for NativeSeriesView {}

#[pymethods]
impl NativeSeriesView {
    fn __len__(&self) -> usize {
        self.base + self.retained
    }

    /// Element shape (without the time axis).
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.extents.clone()
    }

    /// Values in logical `[start, end)` as a `(end-start, *element_shape)` NumPy
    /// array. Indices are clamped to the retained window `[base, len)`, so for a
    /// retention-bounded series the default `values()` returns the kept tail.
    #[pyo3(signature = (start=0, end=None))]
    fn values<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: Option<usize>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let n = self.base + self.retained;
        let start = start.clamp(self.base, n);
        let end = end.unwrap_or(n).clamp(start, n);
        // SAFETY: the cell outlives this call and [start, end) ⊆ [base, n).
        let flat = unsafe {
            std::slice::from_raw_parts(
                self.values.add((start - self.base) * self.stride),
                (end - start) * self.stride,
            )
        };
        let mut full = vec![end - start];
        full.extend_from_slice(&self.extents);
        let nd = ArrayD::from_shape_vec(IxDyn(&full), flat.to_vec()).expect("series shape mismatch");
        PyArrayDyn::from_owned_array(py, nd)
    }

    /// Most recent element as an `element_shape` NumPy array.
    fn last<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        if self.retained == 0 {
            return Err(PyIndexError::new_err("last() on empty series"));
        }
        self.at(py, (self.base + self.retained) as isize - 1)
    }

    /// Element at logical index `i` (supports negative indexing). Raises
    /// `IndexError` if the index has been dropped by the retention bound.
    fn at<'py>(&self, py: Python<'py>, i: isize) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let n = (self.base + self.retained) as isize;
        let idx = if i < 0 { n + i } else { i };
        if idx < 0 || idx >= n {
            return Err(PyIndexError::new_err(format!("index {i} out of bounds (len {n})")));
        }
        let idx = idx as usize;
        if idx < self.base {
            return Err(PyIndexError::new_err(format!(
                "index {i} evicted from retained window [{}, {n})",
                self.base
            )));
        }
        // SAFETY: the cell outlives this call and idx ∈ [base, n).
        let elem = unsafe {
            std::slice::from_raw_parts(self.values.add((idx - self.base) * self.stride), self.stride)
        };
        let nd =
            ArrayD::from_shape_vec(IxDyn(&self.extents), elem.to_vec()).expect("series shape mismatch");
        Ok(PyArrayDyn::from_owned_array(py, nd))
    }

    /// Timestamps in logical `[start, end)` as an int64 (TAI ns) NumPy array.
    /// Indices are clamped to the retained window `[base, len)`.
    #[pyo3(signature = (start=0, end=None))]
    fn slice<'py>(&self, py: Python<'py>, start: usize, end: Option<usize>) -> Bound<'py, PyArray1<i64>> {
        let n = self.base + self.retained;
        let start = start.clamp(self.base, n);
        let end = end.unwrap_or(n).clamp(start, n);
        // SAFETY: the cell outlives this call and [start, end) ⊆ [base, n).
        let window = unsafe {
            std::slice::from_raw_parts(self.timestamps.add(start - self.base), end - start)
        };
        let ts: Vec<i64> = window.iter().map(|t| t.as_nanos()).collect();
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
        let n = self.base + self.retained;
        let ind = sl.indices(n as isize)?;
        if ind.step != 1 {
            return Err(PyValueError::new_err("only contiguous (step 1) slices supported"));
        }
        Ok(self.values(py, ind.start as usize, Some(ind.stop as usize)).into_any())
    }
}

impl NativeSeriesView {
    /// Bind a view over a `Series<f64, N>` cell. The concrete rank `N` is
    /// known at the call site, so the static extents and the retained window's
    /// raw parts are read here; the pyclass itself is rank-erased.
    fn bind<'py, const N: usize>(py: Python<'py>, s: &Series<f64, N>) -> PyResult<Bound<'py, PyAny>> {
        let view = NativeSeriesView {
            values: s.values().as_ptr(),
            timestamps: s.timestamps().as_ptr(),
            retained: s.retained_len(),
            base: s.base(),
            stride: s.stride(),
            extents: s.extents().to_vec(),
        };
        Ok(Bound::new(py, view)?.into_any())
    }
}

// ===========================================================================
// PyArgs — build the Python view tuple + produced bools from typed input refs
// ===========================================================================

/// Walks an operator's input [`Interface`] refs tree, appending one Python view
/// per leaf ([`NativeArrayView`] / [`NativeSeriesView`] / `None`) and one
/// produced (notify) bool per leaf, in tree order.
pub trait PyArgs: Interface + InterfaceHandles {
    /// Append one Python view per leaf to `views` and one notify bit per leaf
    /// to `produced` (views order = legacy input order).
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()>;
}

impl<const N: usize> PyArgs for RefPort<Array<f64, N>> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (notify, value) = refs;
        views.push(NativeArrayView::bind::<N>(
            py,
            value as *const Array<f64, N> as *mut Array<f64, N>,
            false,
        )?);
        produced.push(notify);
        Ok(())
    }
}

impl<const N: usize> PyArgs for RefPort<Series<f64, N>> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (notify, value) = refs;
        views.push(NativeSeriesView::bind::<N>(py, value)?);
        produced.push(notify);
        Ok(())
    }
}

impl PyArgs for RefPort<()> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (notify, _) = refs;
        views.push(py.None().into_bound(py));
        produced.push(notify);
        Ok(())
    }
}

/// A runtime-length group appends one view + bit per element. (Concrete per
/// leaf type — a generic `RefPorts<T> where RefPort<T>: PyArgs` impl cannot pass a
/// `(bool, &T)` tuple where the unnormalized `<RefPort<T> as Interface>::Refs`
/// projection is expected.)
impl<const N: usize> PyArgs for RefPorts<Array<f64, N>> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (flags, values) = refs;
        debug_assert!(flags.len() == values.len(), "RefPorts refs planes disagree on length");
        for (i, &value) in values.iter().enumerate() {
            <RefPort<Array<f64, N>> as PyArgs>::append_views(py, (flags[i], value), views, produced)?;
        }
        Ok(())
    }
}

impl<const N: usize> PyArgs for RefPorts<Series<f64, N>> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (flags, values) = refs;
        debug_assert!(flags.len() == values.len(), "RefPorts refs planes disagree on length");
        for (i, &value) in values.iter().enumerate() {
            <RefPort<Series<f64, N>> as PyArgs>::append_views(py, (flags[i], value), views, produced)?;
        }
        Ok(())
    }
}

macro_rules! tuple_pyargs {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: PyArgs,)+> PyArgs for ($($T,)+) {
            fn append_views<'py>(
                py: Python<'py>,
                refs: Self::Values<'_>,
                views: &mut Vec<Bound<'py, PyAny>>,
                produced: &mut Vec<bool>,
            ) -> PyResult<()> {
                $( $T::append_views(py, refs.$idx, views, produced)?; )+
                Ok(())
            }
        }
    };
}

tuple_pyargs!(0: A);
tuple_pyargs!(0: A, 1: B);
tuple_pyargs!(0: A, 1: B, 2: C);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

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

/// A class-based Python operator over input ports `I` (e.g. `RefPorts<Array<f64,
/// 1>>` or `(RefPort<Array<f64, 1>>, RefPort<Series<f64>>, RefPort<Series<f64>>)`)
/// producing a rank-`NO` array output. Load its definition from a `.py` file, an
/// importable module, or an inline source; each defines `build(**kwargs)` (called
/// with [`PyParams`]) or binds `__op__`.
///
/// [array-view-refactor] `NO` is the static output rank, defaulting to `1` to
/// match the ergonomic
/// [`ScenarioExt::add_py_class_operator`](crate::ScenarioExt::add_py_class_operator)
/// convenience (the strategy operators all emit `(N,)` predictions / weights).
/// The output's element shape comes from `out_shape` (its product must equal the
/// rank-`NO` extents' product); the NumPy boundary negotiates by element count.
pub struct PyClassOperator<I: PyArgs = RefPorts<Array<f64, 1>>, const NO: usize = 1> {
    loader: Loader,
    params: PyParams,
    out_shape: Vec<usize>,
    clock: Clock,
    _marker: PhantomData<fn() -> I>,
}

impl<I: PyArgs, const NO: usize> PyClassOperator<I, NO> {
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

/// State: the deferred config (consumed on the `init` build call), the Python
/// operator instance + its Python state object (created on that call), the
/// driver clock, and the rank-`NO` output buffer.
pub struct PyClassState<const NO: usize> {
    loader: Option<Loader>,
    params: Option<PyParams>,
    out_shape: Vec<usize>,
    operator: Option<Py<PyAny>>,
    py_state: Option<Py<PyAny>>,
    clock: Clock,
    out: Array<f64, NO>,
}

impl<I: PyArgs + 'static, const NO: usize> Operator for PyClassOperator<I, NO> {
    type Inputs = I;
    type Outputs = RefPort<Array<f64, NO>>;
    type State = PyClassState<NO>;

    fn init(self) -> PyClassState<NO> {
        PyClassState {
            loader: Some(self.loader),
            params: Some(self.params),
            out_shape: self.out_shape,
            operator: None,
            py_state: None,
            clock: self.clock,
            out: Array::zeros([0; NO]),
        }
    }

    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut PyClassState<NO>,
        init: bool,
    ) -> (bool, &'a Array<f64, NO>) {
        // The build call runs before the driver's first batch, when the clock
        // is still unset — Python `init` then sees `i64::MIN` (the legacy
        // "no time yet" sentinel).
        let ts = state.clock.get().unwrap_or(Instant::MIN).as_nanos();

        if init {
            // Build call: instantiate the Python operator and call its `init`
            // with the build-time input views (no produced bits — the legacy
            // Python `init(inputs, ts)` contract); allocate the output buffer.
            // No Python `compute` runs here.
            let loader = state.loader.take().expect("build call ran twice");
            let params = state.params.take().expect("build call ran twice");
            let (operator, py_state) = Python::attach(|py| {
                let run = || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                    let operator = resolve_operator(py, &loader, &params)?;
                    let mut views: Vec<Bound<'_, PyAny>> = Vec::new();
                    let mut produced: Vec<bool> = Vec::new(); // discarded on init
                    I::append_views(py, inputs.clone(), &mut views, &mut produced)?;
                    let st = operator.call_method1("init", (PyTuple::new(py, views)?, ts))?;
                    Ok((operator.unbind(), st.unbind()))
                };
                run().unwrap_or_else(|e| {
                    e.print(py);
                    panic!("python operator init failed (see traceback above)");
                })
            });
            state.operator = Some(operator);
            state.py_state = Some(py_state);
            state.out = Array::zeros(out_extents::<NO>(&state.out_shape));
            return (false, &state.out);
        }

        let PyClassState {
            operator,
            py_state,
            out,
            ..
        } = state;
        let operator = operator
            .as_ref()
            .expect("python operator is instantiated on the build call");
        let py_state = py_state
            .as_ref()
            .expect("python operator is instantiated on the build call");
        let out_ptr: *mut Array<f64, NO> = &mut *out;

        let result: Result<bool, ()> = Python::attach(|py| {
            let run = || -> PyResult<bool> {
                let mut views: Vec<Bound<'_, PyAny>> = Vec::new();
                let mut bits: Vec<bool> = Vec::new();
                I::append_views(py, inputs.clone(), &mut views, &mut bits)?;
                let out_view = NativeArrayView::bind::<NO>(py, out_ptr, true)?;
                let op = operator.bind(py);
                op.call_method1(
                    "compute",
                    (
                        py_state.bind(py),
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
        let notify = result
            .unwrap_or_else(|()| panic!("python operator compute failed (see traceback above)"));
        (notify, &*out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b PyClassState<NO>,
    ) -> (bool, &'a Array<f64, NO>) {
        (false, &state.out)
    }
}

/// Resolve the rank-`NO` output extents from the user-supplied element
/// `out_shape`. A shape of the right rank is taken verbatim; otherwise (the
/// common `NO == 1` case, where strategy operators pass `vec![n]`) the shape is
/// flattened to its element-count product along axis 0.
fn out_extents<const NO: usize>(out_shape: &[usize]) -> [usize; NO] {
    if let Ok(ext) = <[usize; NO]>::try_from(out_shape) {
        return ext;
    }
    let total: usize = out_shape.iter().product();
    let mut ext = [1usize; NO];
    if NO > 0 {
        ext[0] = total;
    } else {
        assert_eq!(
            total, 1,
            "PyClassOperator: scalar output (NO=0) requires a single-element out_shape, got {out_shape:?}",
        );
    }
    ext
}

// ===========================================================================
// Tests
// ===========================================================================

// ===========================================================================
// Constructors
// ===========================================================================

/// A class-based Python operator loaded from an importable **module** (on the
/// embedded interpreter's path). The input port tree `I` is inferred from the
/// wiring; `out_shape` is the output element shape (`[]` for a scalar).
pub fn py_class_operator<I: PyArgs, const NO: usize>(
    module: impl Into<String>,
    params: PyParams,
    out_shape: Vec<usize>,
    clock: &Clock,
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_module(module, params, out_shape, clock.clone())
}

/// [`py_class_operator`] from an inline Python **program** binding `__op__`.
pub fn py_class_operator_source<I: PyArgs, const NO: usize>(
    source: impl Into<String>,
    params: PyParams,
    out_shape: Vec<usize>,
    clock: &Clock,
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_source(source, params, out_shape, clock.clone())
}

/// [`py_class_operator`] from a plain `.py` **file** on disk (read now).
pub fn py_class_operator_file<I: PyArgs, const NO: usize>(
    path: impl AsRef<std::path::Path>,
    params: PyParams,
    out_shape: Vec<usize>,
    clock: &Clock,
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_file(path, params, out_shape, clock.clone())
}

#[cfg(test)]
mod tests {
    use super::{PyClassOperator, PyParams};
    use flowgraph::core::Pool;
    use flowgraph::typed::{Builder, RefPort, RefPorts, RefSource};

    use crate::Instant;
    use crate::operators::{AsView, Clock, Record};
    use crate::{Array, Series};

    /// A small stateful operator over one Array input (L1 turnover), used here
    /// purely as a from_source `PyClassOperator` fixture. Raw string at column 0
    /// preserves Python indentation (Rust `\`-continuation would strip it).
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
        let mut b = Builder::new();
        let src = b.push_source(RefSource::new(Array::from_vec([2], vec![0.5_f64, 0.5])));
        let out = b.push(
            // Output is a scalar (`vec![]`), so NO = 0.
            PyClassOperator::<RefPorts<Array<f64, 1>>, 0>::from_source(
                TURNOVER,
                PyParams::new(),
                vec![],
                clock.clone(),
            ),
            &[*src][..],
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        *g.state_mut(src) = Array::from_vec([2], vec![0.5, 0.5]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[0.0]); // warmup

        *g.state_mut(src) = Array::from_vec([2], vec![0.3, 0.7]);
        g.stabilize(&mut pool);
        assert!((g.ref_view(out).as_slice()[0] - 0.4).abs() < 1e-12);

        *g.state_mut(src) = Array::from_vec([2], vec![1.0, 0.0]);
        g.stabilize(&mut pool);
        assert!((g.ref_view(out).as_slice()[0] - 1.4).abs() < 1e-12);
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
        let mut b = Builder::new();
        // weights: Array(2); feed_data: Array(2) recorded into a Series(2).
        let weights = b.push_source(RefSource::new(Array::from_vec([2], vec![1.0_f64, 1.0])));
        let feed = b.push_source(RefSource::new(Array::from_vec([2], vec![0.0_f64, 0.0])));
        // Record consumes the view currency; bridge the source via `AsView`.
        let feed_v = b.push(AsView::<f64, 1>::new(), *feed);
        let series = b.push(Record::new(clock.clone()), feed_v);
        let out = b.push(
            // Scalar output (`vec![]`), so NO = 0.
            PyClassOperator::<(RefPort<Array<f64, 1>>, RefPort<Series<f64, 1>>), 0>::from_source(
                HIST_DOT,
                PyParams::new(),
                vec![],
                clock.clone(),
            ),
            (*weights, series),
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        // Tick 1 @ t=100: feed [1,2]; series=[[1,2]]; dot with [1,1]=3; mean=3.
        clock.set(Instant::from_nanos(100));
        *g.state_mut(weights) = Array::from_vec([2], vec![1.0, 1.0]);
        *g.state_mut(feed) = Array::from_vec([2], vec![1.0, 2.0]);
        g.stabilize(&mut pool);
        assert!((g.ref_view(out).as_slice()[0] - 3.0).abs() < 1e-12);

        // Tick 2 @ t=200: feed [3,4]; series=[[1,2],[3,4]]; dots=3,7; mean=5.
        clock.set(Instant::from_nanos(200));
        *g.state_mut(feed) = Array::from_vec([2], vec![3.0, 4.0]);
        g.stabilize(&mut pool);
        assert!((g.ref_view(out).as_slice()[0] - 5.0).abs() < 1e-12);
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
        let mut b = Builder::new();
        let src = b.push_source(RefSource::new(Array::from_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0])));
        let out = b.push(
            // Scalar output (`vec![]`), so NO = 0.
            PyClassOperator::<RefPorts<Array<f64, 1>>, 0>::from_file(
                &path,
                PyParams::new().float("scale", 3.0),
                vec![],
                clock.clone(),
            ),
            &[*src][..],
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        *g.state_mut(src) = Array::from_vec([4], vec![1.0, 2.0, 3.0, 4.0]);
        g.stabilize(&mut pool);
        assert!((g.ref_view(out).as_slice()[0] - 30.0).abs() < 1e-12); // sum(1..4)=10 * 3
    }

    // -- Integration with the real `flowops` package (needs python/ on -----------
    //    PYTHONPATH + numpy on the embedded interpreter's path). -----------------

    /// A real predictor (LinearRegression) end-to-end: Array universe + two
    /// Series-history inputs (features (N,F), target (N)), produced-gated
    /// rebalance, emitting an (N,) prediction. Validates the heterogeneous
    /// Series path through the engine.
    #[test]
    fn flowops_linear_regression_predictor() {
        const N: usize = 3;
        const F: usize = 2;
        let clock = Clock::new();
        let mut b = Builder::new();
        let universe = b.push_source(RefSource::new(Array::from_vec([N], vec![1.0; N])));
        let feat_feed = b.push_source(RefSource::new(Array::<f64, 2>::zeros([N, F])));
        let tgt_feed = b.push_source(RefSource::new(Array::<f64, 1>::zeros([N])));
        // Record consumes the view currency; bridge each source via `AsView`.
        let feat_v = b.push(AsView::<f64, 2>::new(), *feat_feed);
        let tgt_v = b.push(AsView::<f64, 1>::new(), *tgt_feed);
        let feat_series = b.push(Record::new(clock.clone()), feat_v);
        let tgt_series = b.push(Record::new(clock.clone()), tgt_v);
        let pred = b.push(
            // Output is the (N,) prediction → NO = 1 (the default).
            PyClassOperator::<(RefPort<Array<f64, 1>>, RefPort<Series<f64, 2>>, RefPort<Series<f64, 1>>)>::from_module(
                "flowops.predictors.mean.linear_regression",
                PyParams::new()
                    .int("num_stocks", N as i64)
                    .int("num_features", F as i64)
                    .int("universe_size", N as i64)
                    .int("target_offset", 1),
                vec![N],
                clock.clone(),
            ),
            (*universe, feat_series, tgt_series),
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        // Feed a few ticks of features/targets with a linear relationship so the
        // pooled OLS is well-posed; rebalance each tick (universe produces).
        for t in 1..=5_i64 {
            let x: Vec<f64> = (0..N * F).map(|k| (t as f64) + 0.1 * k as f64).collect();
            let y: Vec<f64> = (0..N).map(|i| 0.5 * (t as f64) + i as f64).collect();
            clock.set(Instant::from_nanos(t * 100));
            *g.state_mut(feat_feed) = Array::from_vec([N, F], x);
            *g.state_mut(tgt_feed) = Array::from_vec([N], y);
            *g.state_mut(universe) = Array::from_vec([N], vec![1.0; N]);
            g.stabilize(&mut pool);
        }

        let mu = g.ref_view(pred).as_slice();
        assert_eq!(mu.len(), N);
        assert!(mu.iter().all(|v| v.is_finite()), "prediction has non-finite entries: {mu:?}");
    }
}
