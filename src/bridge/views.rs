//! Python-visible views for Array, Series, and Notify.
//!
//! - [`NativeArrayView`] — copy-in / copy-out view of a Rust `Array<T>` node.
//! - [`NativeSeriesView`] — copy-out view (plus `push`) of a Rust `Series<T>`
//!   node.
//! - [`NativeNotify`] — wrapper exposing the Rust
//!   [`Notify`](crate::operator::Notify) context to Python operators.
//!
//! All views hold raw pointers into graph-owned memory.  Reads copy data out
//! and writes copy data in, so no reference to graph memory is ever exposed
//! to Python.

use numpy::ndarray::{Array1, ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::{Array, Scalar, Series};

/// Type alias for arbitrary Python objects.
pub type PyObject = Py<PyAny>;

use super::dispatch::dispatch_dtype;

/// Additional bounds needed for numpy interop.
pub trait PyScalar: Scalar + numpy::Element {}

impl PyScalar for bool {}
impl PyScalar for i8 {}
impl PyScalar for i16 {}
impl PyScalar for i32 {}
impl PyScalar for i64 {}
impl PyScalar for u8 {}
impl PyScalar for u16 {}
impl PyScalar for u32 {}
impl PyScalar for u64 {}
impl PyScalar for f32 {}
impl PyScalar for f64 {}

// ===========================================================================
// View creation (dtype-dispatched)
// ===========================================================================

/// What kind of value a node holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    /// A fixed-shape multidimensional array.
    Array,
    /// A time-indexed append-only series.
    Series,
}

/// Create a Python view ([`NativeArrayView`] or [`NativeSeriesView`]) for a
/// node, dispatching on its kind and dtype.
pub fn create_view(
    py: Python<'_>,
    ptr: *mut u8,
    shape: &[usize],
    dtype: &str,
    kind: ViewKind,
) -> PyResult<PyObject> {
    macro_rules! make_view {
        ($T:ty) => {
            match kind {
                ViewKind::Array => {
                    let v = make_array_view::<$T>(ptr, shape, dtype);
                    Ok(Py::new(py, v)?.into_any())
                }
                ViewKind::Series => {
                    let v = make_series_view::<$T>(ptr, shape, dtype);
                    Ok(Py::new(py, v)?.into_any())
                }
            }
        };
    }
    dispatch_dtype!(dtype, make_view)
}

// ===========================================================================
// ArrayView
// ===========================================================================

/// Type-erased function: copy Array<T> data into a new numpy array.
type ArrayValueFn = unsafe fn(*const u8, Python<'_>, &[usize]) -> PyResult<PyObject>;

/// Type-erased function: copy numpy data into Array<T>.
type ArrayWriteFn = unsafe fn(*mut u8, Python<'_>, &Bound<'_, PyAny>, &[usize]) -> PyResult<()>;

/// Python-visible view of a Rust `Array<T>` node.
///
/// Holds a raw pointer to the graph node's value. Every read ([`value`](Self::value))
/// copies data out; every write ([`write`](Self::write)) copies data in. No
/// reference to graph memory is ever exposed to Python.
///
/// Function pointers are monomorphized at creation time, so methods dispatch
/// without runtime dtype matching.
#[pyclass]
pub struct NativeArrayView {
    ptr: *mut u8,
    shape: Vec<usize>,
    dtype_str: String,
    value_fn: ArrayValueFn,
    write_fn: ArrayWriteFn,
}

unsafe impl Send for NativeArrayView {}
unsafe impl Sync for NativeArrayView {}

/// Create an [`NativeArrayView`] for a node holding `Array<T>`.
fn make_array_view<T: PyScalar>(ptr: *mut u8, shape: &[usize], dtype_str: &str) -> NativeArrayView {
    NativeArrayView {
        ptr,
        shape: shape.to_vec(),
        dtype_str: dtype_str.to_string(),
        value_fn: array_value::<T>,
        write_fn: array_write::<T>,
    }
}

#[pymethods]
impl NativeArrayView {
    /// Copy the array data into a new numpy array.
    fn value<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        unsafe { (self.value_fn)(self.ptr as *const u8, py, &self.shape) }
    }

    /// Overwrite the array data from a numpy array.
    fn write<'py>(&self, py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<()> {
        unsafe { (self.write_fn)(self.ptr, py, value, &self.shape) }
    }

    /// Element shape as a Python tuple.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        Ok(pyo3::types::PyTuple::new(py, &self.shape)?
            .into_any()
            .unbind())
    }

    /// Numpy dtype object.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        Ok(np.call_method1("dtype", (&self.dtype_str,))?.unbind())
    }
}

// -- Numpy interop helper ---------------------------------------------------

use super::dispatch::ContiguousArrayInfo;

// -- Monomorphized helpers for ArrayView ------------------------------------

unsafe fn array_value<T: PyScalar>(
    ptr: *const u8,
    py: Python<'_>,
    shape: &[usize],
) -> PyResult<PyObject> {
    let arr = unsafe { &*(ptr as *const Array<T>) };
    let nd = ArrayD::from_shape_vec(IxDyn(shape), arr.as_slice().to_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
}

unsafe fn array_write<T: PyScalar>(
    ptr: *mut u8,
    _py: Python<'_>,
    value: &Bound<'_, PyAny>,
    shape: &[usize],
) -> PyResult<()> {
    let arr = unsafe { &mut *(ptr as *mut Array<T>) };
    let expected_len = arr.as_slice().len();

    let src = ContiguousArrayInfo::try_from(value)?;
    if src.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "shape mismatch: expected {} elements (shape {:?}), got {} (shape {:?})",
            expected_len,
            shape,
            src.len(),
            src.shape,
        )));
    }

    unsafe { src.clone_to_slice(arr.as_mut_slice()) };
    Ok(())
}

// ===========================================================================
// SeriesView
// ===========================================================================

/// Type-erased function: copy Series<T> last element into a new numpy array.
type SeriesLastFn = unsafe fn(*const u8, Python<'_>, &[usize]) -> PyResult<PyObject>;
/// Type-erased function: copy Series<T> values slice into a new numpy array.
type SeriesValuesFn =
    unsafe fn(*const u8, Python<'_>, &[usize], usize, usize) -> PyResult<PyObject>;
/// Type-erased function: copy Series<T> timestamps slice into a new numpy array.
type SeriesIndexFn = unsafe fn(*const u8, Python<'_>, usize, usize) -> PyResult<PyObject>;
/// Type-erased function: get Series<T> length.
type SeriesLenFn = unsafe fn(*const u8) -> usize;
/// Type-erased function: as-of temporal lookup.
type SeriesAsofFn = unsafe fn(*const u8, Python<'_>, &[usize], i64) -> PyResult<PyObject>;
/// Type-erased function: positional element access.
type SeriesAtFn = unsafe fn(*const u8, Python<'_>, &[usize], usize) -> PyResult<PyObject>;
/// Type-erased function: push an element into Series<T>.
type SeriesPushFn =
    unsafe fn(*mut u8, Python<'_>, &Bound<'_, PyAny>, &[usize], i64) -> PyResult<()>;

/// Python-visible view of a Rust `Series<T>` node.
///
/// Holds a raw pointer to the graph node's value. Read methods copy data out;
/// [`push`](Self::push) appends an element, copying data in. Series buffers
/// can reallocate during graph execution (capacity doubling on append); copying
/// prevents dangling references.
///
/// Function pointers are monomorphized at creation time, so methods dispatch
/// without runtime dtype matching.
#[pyclass]
pub struct NativeSeriesView {
    ptr: *mut u8,
    shape: Vec<usize>,
    dtype_str: String,
    last_fn: SeriesLastFn,
    values_fn: SeriesValuesFn,
    index_fn: SeriesIndexFn,
    len_fn: SeriesLenFn,
    asof_fn: SeriesAsofFn,
    at_fn: SeriesAtFn,
    push_fn: SeriesPushFn,
}

unsafe impl Send for NativeSeriesView {}
unsafe impl Sync for NativeSeriesView {}

/// Create a [`NativeSeriesView`] for a node holding `Series<T>`.
fn make_series_view<T: PyScalar>(
    ptr: *mut u8,
    shape: &[usize],
    dtype_str: &str,
) -> NativeSeriesView {
    NativeSeriesView {
        ptr,
        shape: shape.to_vec(),
        dtype_str: dtype_str.to_string(),
        last_fn: series_last::<T>,
        values_fn: series_values::<T>,
        index_fn: series_index::<T>,
        len_fn: series_len::<T>,
        asof_fn: series_asof::<T>,
        at_fn: series_at::<T>,
        push_fn: series_push::<T>,
    }
}

#[pymethods]
impl NativeSeriesView {
    /// Copy the latest element into a new numpy array.
    fn last<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        unsafe { (self.last_fn)(self.ptr as *const u8, py, &self.shape) }
    }

    /// Copy a slice of values into a new numpy array.
    ///
    /// Returns shape `(end - start, *element_shape)`.
    #[pyo3(signature = (start=0, end=None))]
    fn values<'py>(&self, py: Python<'py>, start: usize, end: Option<usize>) -> PyResult<PyObject> {
        let len = unsafe { (self.len_fn)(self.ptr as *const u8) };
        let end = end.unwrap_or(len);
        unsafe { (self.values_fn)(self.ptr as *const u8, py, &self.shape, start, end) }
    }

    /// Copy a slice of timestamps into a new numpy int64 array.
    #[pyo3(signature = (start=0, end=None))]
    fn slice<'py>(&self, py: Python<'py>, start: usize, end: Option<usize>) -> PyResult<PyObject> {
        let len = unsafe { (self.len_fn)(self.ptr as *const u8) };
        let end = end.unwrap_or(len);
        unsafe { (self.index_fn)(self.ptr as *const u8, py, start, end) }
    }

    /// As-of temporal lookup: most recent element with `ts <= timestamp`.
    ///
    /// Returns `None` if no element satisfies the condition.
    #[pyo3(signature = (timestamp,))]
    fn asof<'py>(&self, py: Python<'py>, timestamp: i64) -> PyResult<PyObject> {
        unsafe { (self.asof_fn)(self.ptr as *const u8, py, &self.shape, timestamp) }
    }

    /// Element at positional index (supports negative indexing).
    #[pyo3(signature = (index,))]
    fn at<'py>(&self, py: Python<'py>, index: i64) -> PyResult<PyObject> {
        let len = unsafe { (self.len_fn)(self.ptr as *const u8) };
        let i = if index < 0 {
            let positive = (-index) as usize;
            if positive > len {
                return Err(PyIndexError::new_err(format!(
                    "index {index} out of bounds (len {len})"
                )));
            }
            len - positive
        } else {
            let i = index as usize;
            if i >= len {
                return Err(PyIndexError::new_err(format!(
                    "index {index} out of bounds (len {len})"
                )));
            }
            i
        };
        unsafe { (self.at_fn)(self.ptr as *const u8, py, &self.shape, i) }
    }

    /// Append an element with the given timestamp.
    fn push<'py>(
        &self,
        py: Python<'py>,
        timestamp: i64,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        unsafe { (self.push_fn)(self.ptr, py, value, &self.shape, timestamp) }
    }

    /// Number of recorded elements.
    fn __len__(&self) -> usize {
        unsafe { (self.len_fn)(self.ptr as *const u8) }
    }

    /// Element shape as a Python tuple.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        Ok(pyo3::types::PyTuple::new(py, &self.shape)?
            .into_any()
            .unbind())
    }

    /// Numpy dtype object.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        Ok(np.call_method1("dtype", (&self.dtype_str,))?.unbind())
    }
}

// -- Monomorphized helpers for SeriesView -----------------------------------

unsafe fn series_last<T: PyScalar>(
    ptr: *const u8,
    py: Python<'_>,
    shape: &[usize],
) -> PyResult<PyObject> {
    let series = unsafe { &*(ptr as *const Series<T>) };
    let values = series.last().unwrap_or(&[]);
    let nd = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
}

unsafe fn series_values<T: PyScalar>(
    ptr: *const u8,
    py: Python<'_>,
    element_shape: &[usize],
    start: usize,
    end: usize,
) -> PyResult<PyObject> {
    let series = unsafe { &*(ptr as *const Series<T>) };
    let n = series.len();
    let start = start.min(n);
    let end = end.min(n);
    let stride: usize = element_shape.iter().product::<usize>();
    let all_values = series.values();
    let slice = &all_values[start * stride..end * stride];
    let count = end - start;
    let mut full_shape = vec![count];
    full_shape.extend_from_slice(element_shape);
    let nd = ArrayD::from_shape_vec(IxDyn(&full_shape), slice.to_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
}

unsafe fn series_index<T: PyScalar>(
    ptr: *const u8,
    py: Python<'_>,
    start: usize,
    end: usize,
) -> PyResult<PyObject> {
    let series = unsafe { &*(ptr as *const Series<T>) };
    let n = series.len();
    let start = start.min(n);
    let end = end.min(n);
    // Wire format is TAI ns (matches numpy naive `datetime64[ns]`
    // arithmetic).  Reinterpret the `Instant` slice as `i64`.
    let slice = crate::time::Instant::as_nanos_slice(&series.timestamps()[start..end]);
    let arr = Array1::from(slice.to_vec());
    Ok(PyArray1::from_owned_array(py, arr).into_any().unbind())
}

unsafe fn series_len<T: PyScalar>(ptr: *const u8) -> usize {
    let series = unsafe { &*(ptr as *const Series<T>) };
    series.len()
}

unsafe fn series_asof<T: PyScalar>(
    ptr: *const u8,
    py: Python<'_>,
    shape: &[usize],
    timestamp: i64,
) -> PyResult<PyObject> {
    let series = unsafe { &*(ptr as *const Series<T>) };
    match series.asof(crate::time::Instant::from_nanos(timestamp)) {
        Some(slice) => {
            let nd = ArrayD::from_shape_vec(IxDyn(shape), slice.to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
        }
        None => Ok(py.None().into_pyobject(py)?.unbind()),
    }
}

unsafe fn series_at<T: PyScalar>(
    ptr: *const u8,
    py: Python<'_>,
    shape: &[usize],
    i: usize,
) -> PyResult<PyObject> {
    let series = unsafe { &*(ptr as *const Series<T>) };
    let slice = series.at(i);
    let nd = ArrayD::from_shape_vec(IxDyn(shape), slice.to_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
}

unsafe fn series_push<T: PyScalar>(
    ptr: *mut u8,
    _py: Python<'_>,
    value: &Bound<'_, PyAny>,
    shape: &[usize],
    timestamp: i64,
) -> PyResult<()> {
    let series = unsafe { &mut *(ptr as *mut Series<T>) };
    let stride: usize = shape.iter().product::<usize>();

    let src = ContiguousArrayInfo::try_from(value)?;
    if src.len() != stride {
        return Err(PyValueError::new_err(format!(
            "push: expected {} elements (shape {:?}), got {} (shape {:?})",
            stride,
            shape,
            src.len(),
            src.shape,
        )));
    }

    let mut buf = vec![T::default(); stride];
    unsafe { src.clone_to_slice(&mut buf) };
    series.push(crate::time::Instant::from_nanos(timestamp), &buf);
    Ok(())
}

// ===========================================================================
// NativeNotify
// ===========================================================================

/// Python-visible wrapper around the Rust [`Notify`](crate::operator::Notify)
/// context.
///
/// Provides [`input_produced`](Self::input_produced) (per-position booleans)
/// and [`produced`](Self::produced) (list of positions) so Python operators
/// can check which inputs produced new output in the current flush cycle.
#[pyclass]
pub struct NativeNotify {
    incoming: *const usize,
    incoming_len: usize,
    num_inputs: usize,
}

unsafe impl Send for NativeNotify {}
unsafe impl Sync for NativeNotify {}

impl NativeNotify {
    /// Construct an empty notify.
    pub fn from_empty() -> Self {
        Self {
            incoming: std::ptr::null(),
            incoming_len: 0,
            num_inputs: 0,
        }
    }

    /// Update pointers from a [`Notify`](crate::operator::Notify) reference.
    ///
    /// # Safety
    ///
    /// The `NativeNotify` must not be accessed after the `Notify` it borrows
    /// from is dropped.
    pub unsafe fn update_from(&mut self, notify: &crate::Notify<'_>) {
        self.incoming = notify.incoming_ptr();
        self.incoming_len = notify.incoming_len();
        self.num_inputs = notify.num_inputs();
    }
}

#[pymethods]
impl NativeNotify {
    /// Returns a list of bools indexed by input position: `True` if
    /// that input produced new output in the current flush cycle.
    ///
    /// Computed on each call from the incoming positions list.
    fn input_produced<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let mut flags = vec![false; self.num_inputs];
        if !self.incoming.is_null() {
            let incoming =
                unsafe { std::slice::from_raw_parts(self.incoming, self.incoming_len) };
            for &pos in incoming {
                if pos < self.num_inputs {
                    flags[pos] = true;
                }
            }
        }
        let list = pyo3::types::PyList::new(py, &flags)?;
        Ok(list.into_any().unbind())
    }

    /// Returns the input positions that produced new output in the
    /// current flush cycle.
    fn produced<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let slice = if self.incoming.is_null() || self.incoming_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.incoming, self.incoming_len) }
        };
        let list = pyo3::types::PyList::new(py, slice)?;
        Ok(list.into_any().unbind())
    }
}
