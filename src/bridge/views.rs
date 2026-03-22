//! Python-visible views for Array and Series nodes.
//!
//! [`StoreView`] exposes Rust `Array<T>` or `Series<T>` nodes to Python
//! through raw pointer + dtype dispatch.

use numpy::ndarray::{Array1, ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::array::Array;
use crate::series::Series;

use super::dispatch::normalise_dtype;

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// NodeKind
// ---------------------------------------------------------------------------

/// What kind of value a node holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// Typed `Array<T>` (Rust and Python operators).
    Array,
    /// `Series<T>` (via Record).
    Series,
}

// ---------------------------------------------------------------------------
// StoreView
// ---------------------------------------------------------------------------

/// Python-visible view of a Rust `Array<T>` or `Series<T>`.
#[pyclass]
pub struct StoreView {
    value_ptr: *const u8,
    kind: NodeKind,
    shape: Vec<usize>,
    dtype_str: String,
}

unsafe impl Send for StoreView {}
unsafe impl Sync for StoreView {}

impl StoreView {
    pub fn new(value_ptr: *const u8, kind: NodeKind, shape: Vec<usize>, dtype_str: String) -> Self {
        Self {
            value_ptr,
            kind,
            shape,
            dtype_str,
        }
    }
}

#[pymethods]
impl StoreView {
    /// Current (last) element as a numpy array.
    #[getter]
    fn last<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match self.kind {
            NodeKind::Array => array_to_numpy(py, self.value_ptr, &self.dtype_str),
            NodeKind::Series => series_last_to_numpy(py, self.value_ptr, &self.dtype_str),
        }
    }

    /// All timestamps as a numpy int64 array (Series only).
    #[getter]
    pub fn index<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match self.kind {
            NodeKind::Array => {
                let arr = Array1::<i64>::default(0);
                Ok(PyArray1::from_owned_array(py, arr).into_any().unbind())
            }
            NodeKind::Series => {
                let series = unsafe { &*(self.value_ptr as *const Series<f64>) };
                let arr = Array1::from(series.timestamps().to_vec());
                Ok(PyArray1::from_owned_array(py, arr).into_any().unbind())
            }
        }
    }

    /// All historical values as a numpy array (Series only).
    #[getter]
    pub fn values<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match self.kind {
            NodeKind::Array => array_to_numpy(py, self.value_ptr, &self.dtype_str),
            NodeKind::Series => {
                series_values_to_numpy(py, self.value_ptr, &self.shape, &self.dtype_str)
            }
        }
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

    /// Number of recorded elements (1 for Array nodes, series length for Series nodes).
    pub fn __len__(&self) -> usize {
        match self.kind {
            NodeKind::Array => 1,
            NodeKind::Series => {
                let series = unsafe { &*(self.value_ptr as *const Series<f64>) };
                series.len()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers — Array
// ---------------------------------------------------------------------------

fn array_to_numpy(py: Python<'_>, value_ptr: *const u8, dtype_str: &str) -> PyResult<PyObject> {
    let dtype = normalise_dtype(dtype_str);
    macro_rules! arr_match {
        ($T:ty) => {{
            let arr = unsafe { &*(value_ptr as *const Array<$T>) };
            let shape: Vec<usize> = arr.shape().to_vec();
            let full_shape = if shape.is_empty() { vec![1] } else { shape };
            let nd = ArrayD::from_shape_vec(IxDyn(&full_shape), arr.as_slice().to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
        }};
    }
    match dtype {
        "float64" => arr_match!(f64),
        "float32" => arr_match!(f32),
        "int64" => arr_match!(i64),
        "int32" => arr_match!(i32),
        "uint64" => arr_match!(u64),
        "uint32" => arr_match!(u32),
        "bool" => arr_match!(u8),
        _ => Err(PyTypeError::new_err(format!(
            "unsupported dtype: {dtype_str}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers — Series
// ---------------------------------------------------------------------------

fn series_last_to_numpy(
    py: Python<'_>,
    value_ptr: *const u8,
    dtype_str: &str,
) -> PyResult<PyObject> {
    let dtype = normalise_dtype(dtype_str);
    macro_rules! last_match {
        ($T:ty) => {{
            let series = unsafe { &*(value_ptr as *const Series<$T>) };
            let values = series.last().unwrap_or(&[]);
            let shape = series.shape();
            let full_shape = if shape.is_empty() {
                vec![1]
            } else {
                shape.to_vec()
            };
            let nd = ArrayD::from_shape_vec(IxDyn(&full_shape), values.to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
        }};
    }
    match dtype {
        "float64" => last_match!(f64),
        "float32" => last_match!(f32),
        "int64" => last_match!(i64),
        "int32" => last_match!(i32),
        "uint64" => last_match!(u64),
        "uint32" => last_match!(u32),
        "bool" => last_match!(u8),
        _ => Err(PyTypeError::new_err(format!(
            "unsupported dtype: {dtype_str}"
        ))),
    }
}

fn series_values_to_numpy(
    py: Python<'_>,
    value_ptr: *const u8,
    element_shape: &[usize],
    dtype_str: &str,
) -> PyResult<PyObject> {
    let dtype = normalise_dtype(dtype_str);
    macro_rules! values_match {
        ($T:ty) => {{
            let series = unsafe { &*(value_ptr as *const Series<$T>) };
            let n = series.len();
            let mut full_shape = vec![n];
            full_shape.extend_from_slice(element_shape);
            let nd = ArrayD::from_shape_vec(IxDyn(&full_shape), series.values().to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArrayDyn::from_owned_array(py, nd).into_any().unbind())
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
