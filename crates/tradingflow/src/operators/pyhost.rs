//! Class-based Python operator host (feature `python`).
//!
//! [`PyClassOperator`] is a graph node whose compute step runs a Python operator
//! object, mirroring the legacy `tradingflow.operator.Operator` contract, so the
//! Python-resident operator layer (predictors / portfolios / stateful
//! metrics) ports nearly verbatim:
//!
//! ```text
//! init(inputs, timestamp) -> state
//! compute(state, inputs, output, timestamp, produced) -> bool   # @staticmethod
//! ```
//!
//! `inputs` is a tuple of **views** — [`NativeArrayView`] for array edges
//! (`ArrayPort`) and [`NativeSeriesView`] for recorded-history windows
//! (`SeriesPort` edges), `None` for unit (clock) inputs. `output` is a
//! writable array view, `timestamp` is naive
//! nanoseconds (the graph's ambient event time), `produced` is a `tuple[bool, ...]`
//! parallel to `inputs`, and `state` is a Python object carried across ticks.
//!
//! The `source` is a Python *program* (statements) executed in the operator's
//! own globals (with `np`/`numpy` pre-injected) that binds the operator instance
//! to the name `__op__`.
//!
//! # Heterogeneous inputs
//!
//! The host speaks the operator library's view currency on both sides: inputs
//! are trees of port leaves (`ArrayPort<f64, N>` / `SeriesPort<f64, N>`, plus
//! `RefPort<()>` for unit clocks), runtime-length groups (`ArrayPorts` /
//! `SeriesPorts`), and tuples, so an operator's input shape is its concrete
//! [`Interface`] type, e.g.
//! `(ArrayPort<f64, 1>, SeriesPort<f64, 2>, SeriesPort<f64, 1>)` for a
//! predictor or `ArrayPorts<f64, 1>` for an all-array operator — and the
//! output is an `ArrayPort<f64, NO>` view of the host's owned buffer. The
//! [`PyArgs`] trait walks the input type's payload tree to build the Python
//! view tuple + produced bools in one pass. (An erased enum input would have
//! to clone growing `Series` each tick — `PyArgs` reads the borrowed views
//! directly instead.)
//!
//! # Data model
//!
//! Copy-based (like the legacy bridge): each view copies the data out to a
//! fresh NumPy array on read (a strided input view is materialized
//! row-major at bind), and `output.write()` copies a NumPy array back.
//! Copies make retention safe and the cost is negligible against the NumPy/SciPy
//! math these operators run.
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
//! packages (and `python/`, for the `tradingflow` operator package) visible at
//! runtime via `PYTHONPATH`. Build and test with:
//!
//! ```console
//! set PYO3_PYTHON=<abs>\.venv\Scripts\python.exe
//! set PATH=<dir containing python3xx.dll>;%PATH%
//! set PYTHONPATH=<repo>\python;<abs>\.venv\Lib\site-packages
//! cargo test --features python operators::pyhost
//! ```
//!
//! For free-threaded instead, swap the venv for a `python3.13t` one (`py install
//! 3.13t-64`) and the dll dir accordingly. In production, point
//! `PYTHONPATH`/`PYTHONHOME` at the deployment environment.
//!
//! # Safety
//!
//! The view pyclasses hold a raw pointer to graph edge data — an input view's
//! backing buffer (or its owned row-major materialization) or the host's
//! output buffer — valid only for the duration of one `compute`/`init` (the
//! payload is borrowed by the engine then). Views are created fresh each call
//! and must not be retained past it. The output array view's pointer carries
//! write provenance (`&mut`); input views are read-only. `unsafe Send + Sync`
//! is sound: a node's `compute` runs on one thread at a time and views never
//! cross threads.

use std::ffi::CString;
use std::marker::PhantomData;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArrayDyn};

use crate::data::{Array, ArrayView, Instant, Layout, SeriesView};
use crate::graph::typed::{Interface, InterfaceHandles, Operator};
use crate::ports::{ArrayPort, ArrayPorts, SeriesPort, SeriesPorts, UnitPort};

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
// is row-major contiguous — `data()`/`data_mut()` over the whole buffer
// are rank-independent. The output view reconstructs the typed `&mut Array<f64,
// NO>` from the stored pointer + rank tag (see [`NativeArrayView::bind`]).

/// View over an array edge's data — an input [`ArrayView`] or the host's
/// output `Array<f64, N>` buffer. `value()` copies out; `write()` copies in
/// (output views only). Valid only during the call that created it.
#[pyclass]
pub struct NativeArrayView {
    /// Pointer to the flat row-major `f64` buffer.
    data: *mut f64,
    /// Number of scalars (product of extents).
    len: usize,
    /// Element extents (the edge's static `[usize; N]`, as a runtime vector).
    extents: Vec<usize>,
    writable: bool,
    /// Owned row-major materialization of a strided input view (see
    /// [`bind_view`](Self::bind_view)); `data` points into it when present.
    /// Never read back — it exists to keep the pointee alive for the call.
    _backing: Option<Vec<f64>>,
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
            return Err(PyValueError::new_err(
                "cannot write to a read-only (input) view",
            ));
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
    /// Bind the writable output view over the host's owned `Array<f64, N>`
    /// buffer. The concrete rank `N` is known at the call site, so the static
    /// extents are read here and stored runtime-wide; the flat buffer is
    /// rank-agnostic.
    fn bind<'py, const N: usize>(
        py: Python<'py>,
        arr: *mut Array<f64, N>,
        writable: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // SAFETY: `arr` is a live cell for the duration of this call (module docs).
        let r = unsafe { &mut *arr };
        let view = NativeArrayView {
            data: r.data_mut().as_mut_ptr(),
            len: r.layout().len(),
            extents: r.extents().to_vec(),
            writable,
            _backing: None,
        };
        Ok(Bound::new(py, view)?.into_any())
    }

    /// Bind a read-only view over an `ArrayPort` input payload (a possibly
    /// strided [`ArrayView`]). A contiguous view is pointed at directly
    /// (zero-copy until Python reads it); a strided one is materialized
    /// row-major into an owned backing the pyclass carries for the call.
    fn bind_view<'py, const N: usize>(
        py: Python<'py>,
        v: ArrayView<'_, f64, N>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let extents = v.extents().to_vec();
        let (data, len, backing) = match v.to_contiguous() {
            std::borrow::Cow::Borrowed(s) => (s.as_ptr().cast_mut(), s.len(), None),
            std::borrow::Cow::Owned(vec) => (vec.as_ptr().cast_mut(), vec.len(), Some(vec)),
        };
        let view = NativeArrayView {
            data,
            len,
            extents,
            writable: false,
            _backing: backing,
        };
        Ok(Bound::new(py, view)?.into_any())
    }
}

// ===========================================================================
// NativeSeriesView — Python-visible view over a cell's `Series<f64>` (history)
// ===========================================================================

/// Read-only view over a recorded-history window (a [`SeriesView`] carried by
/// a `SeriesPort` edge): positional history access with **window-local**
/// indices (`0` is the oldest retained row). Valid only during the call that
/// created it. Rank-erased like [`NativeArrayView`]: the per-leaf [`PyArgs`]
/// impl knows the concrete rank `N` at the bind site and captures the
/// window's raw parts there, so a single non-generic pyclass serves every
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
        let nd =
            ArrayD::from_shape_vec(IxDyn(&full), flat.to_vec()).expect("series shape mismatch");
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
            return Err(PyIndexError::new_err(format!(
                "index {i} out of bounds (len {n})"
            )));
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
            std::slice::from_raw_parts(
                self.values.add((idx - self.base) * self.stride),
                self.stride,
            )
        };
        let nd = ArrayD::from_shape_vec(IxDyn(&self.extents), elem.to_vec())
            .expect("series shape mismatch");
        Ok(PyArrayDyn::from_owned_array(py, nd))
    }

    /// Timestamps in logical `[start, end)` as an int64 (naive ns) NumPy array.
    /// Indices are clamped to the retained window `[base, len)`.
    #[pyo3(signature = (start=0, end=None))]
    fn slice<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: Option<usize>,
    ) -> Bound<'py, PyArray1<i64>> {
        let n = self.base + self.retained;
        let start = start.clamp(self.base, n);
        let end = end.unwrap_or(n).clamp(start, n);
        // SAFETY: the cell outlives this call and [start, end) ⊆ [base, n).
        let window = unsafe {
            std::slice::from_raw_parts(self.timestamps.add(start - self.base), end - start)
        };
        let ts: Vec<i64> = window.iter().map(|t| t.as_offset().as_nanos()).collect();
        PyArray1::from_slice(py, &ts)
    }

    /// Positional indexing: `int` -> single element, contiguous `slice` -> range.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            return Ok(self.at(py, i)?.into_any());
        }
        let sl = key.cast::<PySlice>().map_err(|_| {
            PyValueError::new_err("series index must be an int or a contiguous slice")
        })?;
        let n = self.base + self.retained;
        let ind = sl.indices(n as isize)?;
        if ind.step != 1 {
            return Err(PyValueError::new_err(
                "only contiguous (step 1) slices supported",
            ));
        }
        Ok(self
            .values(py, ind.start as usize, Some(ind.stop as usize))
            .into_any())
    }
}

impl NativeSeriesView {
    /// Bind a view over a recorded-history window ([`SeriesView`]). The
    /// concrete rank `N` is known at the call site, so the static extents and
    /// the window's raw parts are read here; the pyclass itself is
    /// rank-erased. A `SeriesView` carries no logical bookkeeping, so indices
    /// are **window-local** (`base` is always 0: index `0` is the oldest
    /// retained row).
    fn bind<'py, const N: usize>(
        py: Python<'py>,
        s: SeriesView<'_, f64, N>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Python reads the buffer flat, at `base + i * stride + j`, so the
        // window must be packed row-major — which is exactly when a series
        // view yields one slice.
        let values = s.as_slice().ok_or_else(|| {
            PyValueError::new_err("bind: series window is not contiguous row-major")
        })?;
        let view = NativeSeriesView {
            values: values.as_ptr(),
            timestamps: s.timestamps().as_ptr(),
            retained: s.len(),
            base: 0,
            stride: s.layout().len(),
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

impl<const N: usize> PyArgs for ArrayPort<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (notify, value) = refs;
        views.push(NativeArrayView::bind_view::<N>(py, value)?);
        produced.push(notify);
        Ok(())
    }
}

impl<const N: usize> PyArgs for SeriesPort<f64, N> {
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

impl PyArgs for UnitPort {
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
/// leaf type — a generic `ViewPorts<V> where ViewPort<V>: PyArgs` impl cannot
/// pass a `(bool, View)` tuple where the unnormalized
/// `<ViewPort<V> as Interface>::Values` projection is expected.)
impl<const N: usize> PyArgs for ArrayPorts<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (flags, values) = refs;
        debug_assert!(
            flags.len() == values.len(),
            "ArrayPorts payload planes disagree on length"
        );
        for (i, &value) in values.iter().enumerate() {
            <ArrayPort<f64, N> as PyArgs>::append_views(py, (flags[i], value), views, produced)?;
        }
        Ok(())
    }
}

impl<const N: usize> PyArgs for SeriesPorts<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (flags, values) = refs;
        debug_assert!(
            flags.len() == values.len(),
            "SeriesPorts refs planes disagree on length"
        );
        for (i, &value) in values.iter().enumerate() {
            <SeriesPort<f64, N> as PyArgs>::append_views(py, (flags[i], value), views, produced)?;
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

/// A class-based Python operator over input ports `I` (e.g. `ArrayPorts<f64,
/// 1>` or `(ArrayPort<f64, 1>, SeriesPort<f64, 2>, SeriesPort<f64, 1>)`)
/// producing a rank-`NO` array output (an `ArrayPort<f64, NO>` view of the
/// host's owned buffer — the host speaks the view currency on both sides).
/// Load its definition from a `.py` file, an importable module, or an inline
/// source; each defines `build(**kwargs)` (called with [`PyParams`]) or binds
/// `__op__`.
///
/// `NO` is the static output rank, defaulting to `1` because the strategy
/// operators all emit `(N,)` predictions / weights. The output's element shape
/// comes from `out_shape` (its product must equal the rank-`NO` extents'
/// product); the NumPy boundary negotiates by element count.
pub struct PyClassOperator<I: PyArgs = ArrayPorts<f64, 1>, const NO: usize = 1> {
    loader: Loader,
    params: PyParams,
    out_shape: Vec<usize>,
    _marker: PhantomData<fn() -> I>,
}

impl<I: PyArgs, const NO: usize> PyClassOperator<I, NO> {
    /// Load from a `.py` file on disk (read now).
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
        params: PyParams,
        out_shape: Vec<usize>,
    ) -> Self {
        let path = path.as_ref();
        let src = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read python operator file {}: {e}", path.display()));
        Self::from_source(src, params, out_shape)
    }

    /// Load from an importable module (on the embedded interpreter's path).
    pub fn from_module(module: impl Into<String>, params: PyParams, out_shape: Vec<usize>) -> Self {
        Self {
            loader: Loader::Module(module.into()),
            params,
            out_shape,
            _marker: PhantomData,
        }
    }

    /// Load from an inline Python program (handy for tests / one-offs).
    pub fn from_source(source: impl Into<String>, params: PyParams, out_shape: Vec<usize>) -> Self {
        Self {
            loader: Loader::Source(source.into()),
            params,
            out_shape,
            _marker: PhantomData,
        }
    }
}

/// State: the Python operator instance + its Python state object (created at
/// init), and the rank-`NO` output buffer. No clock: the event time arrives as
/// the graph context, per `compute` call.
pub struct PyClassState<const NO: usize> {
    operator: Py<PyAny>,
    py_state: Py<PyAny>,
    out: Array<f64, NO>,
}

impl<I: PyArgs + 'static, const NO: usize> Operator for PyClassOperator<I, NO> {
    type Inputs = I;
    type Outputs = ArrayPort<f64, NO>;
    type Context = Instant;
    type State = PyClassState<NO>;

    fn init(self, inputs: <I as Interface>::Values<'_>) -> PyClassState<NO> {
        // Instantiate the Python operator and call its `init` with the
        // build-time input views (no produced bits — the legacy Python
        // `init(inputs, ts)` contract); allocate the output buffer. No Python
        // `compute` runs here. Init runs before the driver's first batch, so
        // Python `init` sees `i64::MIN` — the "no time yet" sentinel of the
        // legacy contract.
        let ts = Instant::MIN.as_offset().as_nanos();
        let (operator, py_state) = Python::attach(|py| {
            let run = || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                let operator = resolve_operator(py, &self.loader, &self.params)?;
                let mut views: Vec<Bound<'_, PyAny>> = Vec::new();
                let mut produced: Vec<bool> = Vec::new(); // discarded on init
                I::append_views(py, inputs, &mut views, &mut produced)?;
                let st = operator.call_method1("init", (PyTuple::new(py, views)?, ts))?;
                Ok((operator.unbind(), st.unbind()))
            };
            run().unwrap_or_else(|e| {
                e.print(py);
                panic!("python operator init failed (see traceback above)");
            })
        });
        PyClassState {
            operator,
            py_state,
            out: Array::zeros(out_extents::<NO>(&self.out_shape)),
        }
    }

    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut PyClassState<NO>,
        time: &Instant,
    ) -> (bool, ArrayView<'a, f64, NO>) {
        // The batch's event time, straight from the graph context.
        let ts = time.as_offset().as_nanos();

        let PyClassState {
            operator,
            py_state,
            out,
        } = state;
        let out_ptr: *mut Array<f64, NO> = &mut *out;

        let result: Result<bool, ()> = Python::attach(|py| {
            let run = || -> PyResult<bool> {
                let mut views: Vec<Bound<'_, PyAny>> = Vec::new();
                let mut bits: Vec<bool> = Vec::new();
                I::append_views(py, inputs, &mut views, &mut bits)?;
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
        (notify, out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b mut PyClassState<NO>,
    ) -> (bool, ArrayView<'a, f64, NO>) {
        (false, state.out.view())
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
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_module(module, params, out_shape)
}

/// [`py_class_operator`] from an inline Python **program** binding `__op__`.
pub fn py_class_operator_source<I: PyArgs, const NO: usize>(
    source: impl Into<String>,
    params: PyParams,
    out_shape: Vec<usize>,
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_source(source, params, out_shape)
}

/// [`py_class_operator`] from a plain `.py` **file** on disk (read now).
pub fn py_class_operator_file<I: PyArgs, const NO: usize>(
    path: impl AsRef<std::path::Path>,
    params: PyParams,
    out_shape: Vec<usize>,
) -> PyClassOperator<I, NO> {
    PyClassOperator::from_file(path, params, out_shape)
}

#[cfg(test)]
mod tests {
    use super::{PyClassOperator, PyParams};
    use crate::data::{Array, Duration, Instant};
    use crate::graph::pool::Pool;
    use crate::graph::typed::Builder;
    use crate::operators::constant::const_array;
    use crate::operators::structural::Record;
    use crate::ports::{ArrayPort, ArrayPorts, SeriesPort};

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
        let mut b = Builder::new();
        let (src_cell, src) = b.source(const_array(Array::from_parts(
            [2],
            vec![0.5_f64, 0.5].into(),
        )));
        let out = b.segment(
            // Output is a scalar (`vec![]`), so NO = 0.
            PyClassOperator::<ArrayPorts<f64, 1>, 0>::from_source(
                TURNOVER,
                PyParams::new(),
                vec![],
            ),
            &[src][..],
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        *g.state_mut(src_cell) = Array::from_parts([2], vec![0.5, 0.5].into());
        g.stabilize(&mut pool, &Instant::MIN);
        assert_eq!(g.view(out).as_slice().unwrap(), &[0.0]); // warmup

        *g.state_mut(src_cell) = Array::from_parts([2], vec![0.3, 0.7].into());
        g.stabilize(&mut pool, &Instant::MIN);
        assert!((g.view(out).as_slice().unwrap()[0] - 0.4).abs() < 1e-12);

        *g.state_mut(src_cell) = Array::from_parts([2], vec![1.0, 0.0].into());
        g.stabilize(&mut pool, &Instant::MIN);
        assert!((g.view(out).as_slice().unwrap()[0] - 1.4).abs() < 1e-12);
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
        let mut b = Builder::new();
        // weights: Array(2); feed_data: Array(2) recorded into a Series(2).
        // Sources lend the view currency directly — `Record` wires straight on.
        let (weights_cell, weights) = b.source(const_array(Array::from_parts(
            [2],
            vec![1.0_f64, 1.0].into(),
        )));
        let (feed_cell, feed) = b.source(const_array(Array::from_parts(
            [2],
            vec![0.0_f64, 0.0].into(),
        )));
        let series = b.segment(Record::new(), feed);
        let out = b.segment(
            // Scalar output (`vec![]`), so NO = 0.
            PyClassOperator::<(ArrayPort<f64, 1>, SeriesPort<f64, 1>), 0>::from_source(
                HIST_DOT,
                PyParams::new(),
                vec![],
            ),
            (weights, series),
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        // Tick 1 @ t=100: feed [1,2]; series=[[1,2]]; dot with [1,1]=3; mean=3.
        let ctx = Instant::from_offset(Duration::from_nanos(100));
        *g.state_mut(weights_cell) = Array::from_parts([2], vec![1.0, 1.0].into());
        *g.state_mut(feed_cell) = Array::from_parts([2], vec![1.0, 2.0].into());
        g.stabilize(&mut pool, &ctx);
        assert!((g.view(out).as_slice().unwrap()[0] - 3.0).abs() < 1e-12);

        // Tick 2 @ t=200: feed [3,4]; series=[[1,2],[3,4]]; dots=3,7; mean=5.
        let ctx = Instant::from_offset(Duration::from_nanos(200));
        *g.state_mut(feed_cell) = Array::from_parts([2], vec![3.0, 4.0].into());
        g.stabilize(&mut pool, &ctx);
        assert!((g.view(out).as_slice().unwrap()[0] - 5.0).abs() < 1e-12);
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

        let mut b = Builder::new();
        let (src_cell, src) = b.source(const_array(Array::from_parts(
            [4],
            vec![1.0_f64, 2.0, 3.0, 4.0].into(),
        )));
        let out = b.segment(
            // Scalar output (`vec![]`), so NO = 0.
            PyClassOperator::<ArrayPorts<f64, 1>, 0>::from_file(
                &path,
                PyParams::new().float("scale", 3.0),
                vec![],
            ),
            &[src][..],
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        *g.state_mut(src_cell) = Array::from_parts([4], vec![1.0, 2.0, 3.0, 4.0].into());
        g.stabilize(&mut pool, &Instant::MIN);
        // sum(1..4)=10 * 3
        assert!((g.view(out).as_slice().unwrap()[0] - 30.0).abs() < 1e-12);
    }

    // -- Integration with the real `tradingflow` Python package (needs python/ --
    //    on PYTHONPATH + numpy on the embedded interpreter's path). --------------

    /// A real predictor (LinearRegression) end-to-end: Array universe + two
    /// Series-history inputs (features (N,F), target (N)), produced-gated
    /// rebalance, emitting an (N,) prediction. Validates the heterogeneous
    /// Series path through the engine.
    #[test]
    fn pyhost_linear_regression_predictor() {
        const N: usize = 3;
        const F: usize = 2;
        let mut b = Builder::new();
        let (universe_cell, universe) =
            b.source(const_array(Array::from_parts([N], vec![1.0; N].into())));
        let (feat_feed_cell, feat_feed) = b.source(const_array(Array::<f64, 2>::zeros([N, F])));
        let (tgt_feed_cell, tgt_feed) = b.source(const_array(Array::<f64, 1>::zeros([N])));
        // Sources lend the view currency directly — `Record` wires straight on.
        let feat_series = b.segment(Record::new(), feat_feed);
        let tgt_series = b.segment(Record::new(), tgt_feed);
        let pred = b.segment(
            // Output is the (N,) prediction → NO = 1 (the default).
            PyClassOperator::<(ArrayPort<f64, 1>, SeriesPort<f64, 2>, SeriesPort<f64, 1>)>::from_module(
                "tradingflow.predictors.mean.linear_regression",
                PyParams::new()
                    .int("num_stocks", N as i64)
                    .int("num_features", F as i64)
                    .int("universe_size", N as i64)
                    .int("target_offset", 1),
                vec![N],
            ),
            (universe, feat_series, tgt_series),
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        // Feed a few ticks of features/targets with a linear relationship so the
        // pooled OLS is well-posed; rebalance each tick (universe produces).
        for t in 1..=5_i64 {
            let x: Vec<f64> = (0..N * F).map(|k| (t as f64) + 0.1 * k as f64).collect();
            let y: Vec<f64> = (0..N).map(|i| 0.5 * (t as f64) + i as f64).collect();
            let ctx = Instant::from_offset(Duration::from_nanos(t * 100));
            *g.state_mut(feat_feed_cell) = Array::from_parts([N, F], x.into());
            *g.state_mut(tgt_feed_cell) = Array::from_parts([N], y.into());
            *g.state_mut(universe_cell) = Array::from_parts([N], vec![1.0; N].into());
            g.stabilize(&mut pool, &ctx);
        }

        let mu = g.view(pred).as_slice().unwrap();
        assert_eq!(mu.len(), N);
        assert!(
            mu.iter().all(|v| v.is_finite()),
            "prediction has non-finite entries: {mu:?}"
        );
    }
}
