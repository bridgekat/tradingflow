//! Python-visible store views.
//!
//! [`StoreView`] exposes a Rust [`Store<T>`](crate::store::Store) to Python
//! through raw pointer + dtype dispatch.  It provides both element-level
//! access (`.last`, `.shape`) and series-level access (`.index`, `.values`)
//! from a single pyclass.

use numpy::ndarray::{Array1, ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::store::Store;

use super::dispatch::normalise_dtype;

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// StoreView
// ---------------------------------------------------------------------------

/// Python-visible view of a Rust `Store<T>`.
///
/// Wraps a raw pointer to the store plus metadata (dtype, element shape,
/// stride).  All accessors dispatch on `dtype_str` to recover the concrete
/// `Store<T>` type.
#[pyclass]
pub struct StoreView {
    store_ptr: *const u8,
    shape: Vec<usize>,
    #[allow(dead_code)]
    stride: usize,
    dtype_str: String,
}

// SAFETY: StoreView is only accessed from a single Python thread at a time.
// The underlying Store is owned by the Scenario and outlives the view.
unsafe impl Send for StoreView {}
unsafe impl Sync for StoreView {}

impl StoreView {
    /// Create a new StoreView for a node.
    pub fn new(store_ptr: *const u8, shape: Vec<usize>, stride: usize, dtype_str: String) -> Self {
        Self {
            store_ptr,
            shape,
            stride,
            dtype_str,
        }
    }
}

#[pymethods]
impl StoreView {
    /// Current (last) element as a numpy array.
    #[getter]
    fn last<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        store_current_to_numpy(py, self.store_ptr, &self.dtype_str)
    }

    /// All timestamps as a numpy int64 array.
    #[getter]
    pub fn index<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        // All Store<T> have the same timestamps layout; use f64 to access.
        let store = unsafe { &*(self.store_ptr as *const Store<f64>) };
        let arr = Array1::from(store.timestamps().to_vec());
        Ok(PyArray1::from_owned_array(py, arr).into_any().unbind())
    }

    /// All historical values as a numpy array.
    #[getter]
    pub fn values<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        store_values_to_numpy(py, self.store_ptr, &self.shape, &self.dtype_str)
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

    /// Number of elements in the store history.
    pub fn __len__(&self) -> usize {
        // All Store<T> have the same timestamps layout.
        let store = unsafe { &*(self.store_ptr as *const Store<f64>) };
        store.len()
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Read a Store's current element and return as numpy array.
fn store_current_to_numpy(
    py: Python<'_>,
    store_ptr: *const u8,
    dtype_str: &str,
) -> PyResult<PyObject> {
    let dtype = normalise_dtype(dtype_str);
    macro_rules! current_match {
        ($T:ty) => {{
            let store = unsafe { &*(store_ptr as *const Store<$T>) };
            let view = store.current_view();
            let shape: Vec<usize> = view.shape.to_vec();
            let full_shape = if shape.is_empty() { vec![1] } else { shape };
            let arr = ArrayD::from_shape_vec(IxDyn(&full_shape), view.values.to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }};
    }
    match dtype {
        "float64" => current_match!(f64),
        "float32" => current_match!(f32),
        "int64" => current_match!(i64),
        "int32" => current_match!(i32),
        "uint64" => current_match!(u64),
        "uint32" => current_match!(u32),
        "bool" => current_match!(u8),
        _ => Err(PyTypeError::new_err(format!(
            "unsupported dtype: {dtype_str}"
        ))),
    }
}

/// Read a Store's full values and return as numpy array.
fn store_values_to_numpy(
    py: Python<'_>,
    store_ptr: *const u8,
    element_shape: &[usize],
    dtype_str: &str,
) -> PyResult<PyObject> {
    let dtype = normalise_dtype(dtype_str);
    macro_rules! values_match {
        ($T:ty) => {{
            let store = unsafe { &*(store_ptr as *const Store<$T>) };
            let sv = store.series_view();
            let n = sv.len();
            let mut full_shape = vec![n];
            full_shape.extend_from_slice(element_shape);
            let arr = ArrayD::from_shape_vec(IxDyn(&full_shape), sv.values.to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArrayDyn::from_owned_array(py, arr).into_any().unbind())
        }};
    }
    match dtype {
        "float64" => values_match!(f64),
        "float32" => values_match!(f32),
        "int64" => values_match!(i64),
        "int32" => values_match!(i32),
        "uint64" => values_match!(u64),
        "uint32" => values_match!(u32),
        "bool" => values_match!(u8),
        _ => Err(PyTypeError::new_err(format!(
            "unsupported dtype: {dtype_str}"
        ))),
    }
}
