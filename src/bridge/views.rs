//! Python-visible views for Array and Series nodes.
//!
//! [`_ArrayView`] and [`_SeriesView`] are `#[pyclass]` wrappers that hold a raw
//! pointer to a graph node's value.  All read/write methods **copy** data
//! across the boundary — no reference to graph memory ever reaches Python.
//!
//! Each view stores pre-resolved function pointers (monomorphized at creation
//! time), so methods dispatch without runtime dtype matching.
//!
//! # Safety
//!
//! The raw pointer is valid for the lifetime of the owning [`Scenario`].  Since
//! all methods copy (never expose a reference), Python code cannot create
//! dangling pointers even if the underlying buffer is reallocated (e.g. by
//! `mem::replace` on an Array, or capacity-doubling on a Series).

use numpy::ndarray::{Array1, ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::{Array, Scalar, Series};

/// Type alias for arbitrary Python objects.
pub type PyObject = Py<PyAny>;

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
// ArrayView
// ===========================================================================

/// Type-erased function: copy Array<T> data into a new numpy array.
type ArrayValueFn = unsafe fn(*const u8, Python<'_>, &[usize]) -> PyResult<PyObject>;

/// Type-erased function: copy numpy data into Array<T>.
type ArrayWriteFn = unsafe fn(*mut u8, Python<'_>, &Bound<'_, PyAny>, &[usize]) -> PyResult<()>;

/// Python-visible view of a Rust `Array<T>` node.
///
/// Every read (`value`) copies data out.  Every write (`write`) copies data
/// in.  No reference to graph memory is ever exposed to Python.
#[pyclass]
pub struct _ArrayView {
    ptr: *mut u8,
    shape: Vec<usize>,
    dtype_str: String,
    value_fn: ArrayValueFn,
    write_fn: ArrayWriteFn,
}

unsafe impl Send for _ArrayView {}
unsafe impl Sync for _ArrayView {}

/// Create an [`_ArrayView`] for a node holding `Array<T>`.
pub fn make_array_view<T: PyScalar>(ptr: *mut u8, shape: &[usize], dtype_str: &str) -> _ArrayView {
    _ArrayView {
        ptr,
        shape: shape.to_vec(),
        dtype_str: dtype_str.to_string(),
        value_fn: array_value::<T>,
        write_fn: array_write::<T>,
    }
}

#[pymethods]
impl _ArrayView {
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

// -- Monomorphized helpers for ArrayView ------------------------------------

unsafe fn array_value<T: PyScalar>(
    ptr: *const u8,
    py: Python<'_>,
    shape: &[usize],
) -> PyResult<PyObject> {
    let arr = unsafe { &*(ptr as *const Array<T>) };
    let full_shape = if shape.is_empty() {
        vec![1]
    } else {
        shape.to_vec()
    };
    let nd = ArrayD::from_shape_vec(IxDyn(&full_shape), arr.as_slice().to_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
}

unsafe fn array_write<T: PyScalar>(
    ptr: *mut u8,
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    shape: &[usize],
) -> PyResult<()> {
    let arr = unsafe { &mut *(ptr as *mut Array<T>) };
    let expected_len = arr.as_slice().len();

    let np = py.import("numpy")?;
    let contiguous = np.call_method1("ascontiguousarray", (value,))?;
    let interface = contiguous.getattr("__array_interface__")?;
    let data_tuple = interface.get_item("data")?;
    let src_ptr: usize = data_tuple.get_item(0)?.extract()?;

    let src_shape: Vec<usize> = interface.get_item("shape")?.extract()?;
    let src_len: usize = if src_shape.is_empty() {
        1
    } else {
        src_shape.iter().product()
    };
    if src_len != expected_len {
        return Err(PyValueError::new_err(format!(
            "shape mismatch: expected {} elements (shape {:?}), got {} (shape {:?})",
            expected_len, shape, src_len, src_shape,
        )));
    }

    let dst = arr.as_slice_mut();
    unsafe {
        std::ptr::copy_nonoverlapping(src_ptr as *const T, dst.as_mut_ptr(), expected_len);
    }
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

/// Python-visible view of a Rust `Series<T>` node.
///
/// Every read method copies data out.  Series buffers can reallocate during
/// graph execution (capacity doubling on append); copies prevent dangling.
#[pyclass]
pub struct _SeriesView {
    ptr: *mut u8,
    shape: Vec<usize>,
    dtype_str: String,
    last_fn: SeriesLastFn,
    values_fn: SeriesValuesFn,
    index_fn: SeriesIndexFn,
    len_fn: SeriesLenFn,
    asof_fn: SeriesAsofFn,
    at_fn: SeriesAtFn,
}

unsafe impl Send for _SeriesView {}
unsafe impl Sync for _SeriesView {}

/// Create a [`_SeriesView`] for a node holding `Series<T>`.
pub fn make_series_view<T: PyScalar>(
    ptr: *mut u8,
    shape: &[usize],
    dtype_str: &str,
) -> _SeriesView {
    _SeriesView {
        ptr,
        shape: shape.to_vec(),
        dtype_str: dtype_str.to_string(),
        last_fn: series_last::<T>,
        values_fn: series_values::<T>,
        index_fn: series_index::<T>,
        len_fn: series_len::<T>,
        asof_fn: series_asof::<T>,
        at_fn: series_at::<T>,
    }
}

#[pymethods]
impl _SeriesView {
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
    let full_shape = if shape.is_empty() {
        vec![1]
    } else {
        shape.to_vec()
    };
    let nd = ArrayD::from_shape_vec(IxDyn(&full_shape), values.to_vec())
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
    let stride: usize = if element_shape.is_empty() {
        1
    } else {
        element_shape.iter().product()
    };
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
    let slice = &series.timestamps()[start..end];
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
    match series.asof(timestamp) {
        Some(slice) => {
            let full_shape = if shape.is_empty() {
                vec![1]
            } else {
                shape.to_vec()
            };
            let nd = ArrayD::from_shape_vec(IxDyn(&full_shape), slice.to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
        }
        None => Ok(py.None().into_pyobject(py)?.unbind().into()),
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
    let full_shape = if shape.is_empty() {
        vec![1]
    } else {
        shape.to_vec()
    };
    let nd = ArrayD::from_shape_vec(IxDyn(&full_shape), slice.to_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
}
