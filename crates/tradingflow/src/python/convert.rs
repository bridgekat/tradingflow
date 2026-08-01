use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{Element, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::data::{Array, ArrayView, Scalar};

/// Constructs a NumPy array by copying from an [`ArrayView`].
pub fn to_numpy<'py, T: Scalar + Element, const N: usize>(
    py: Python<'py>,
    v: ArrayView<'_, T, N>,
) -> Bound<'py, PyArrayDyn<T>> {
    let nd = ArrayD::from_shape_vec(IxDyn(&v.extents()), v.to_contiguous().into_owned())
        .expect("layout guarantees shape/len agreement");
    PyArrayDyn::from_owned_array(py, nd)
}

/// Constructs an [`Array`] by copying from a NumPy array.
pub fn from_numpy<T: Scalar + Element, const N: usize>(
    value: &Bound<'_, PyAny>,
) -> PyResult<Array<T, N>> {
    let arr = to_contiguous::<T>(value)?;
    let readonly = arr.extract::<PyReadonlyArrayDyn<'_, T>>()?;
    let src = readonly
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("new_from_numpy: non-contiguous value: {e}")))?;

    let shape = readonly.shape();
    let extents: [usize; N] = if shape.len() == N {
        std::array::from_fn(|d| shape[d])
    } else if N == 1 {
        [src.len(); N]
    } else if N == 0 && src.len() == 1 {
        [0; N]
    } else {
        return Err(PyValueError::new_err(format!(
            "new_from_numpy: cannot take rank-{N} extents from shape {shape:?}",
        )));
    };
    let mut out = Array::zeros(extents);
    out.data_mut().clone_from_slice(src);

    Ok(out)
}

/// Copies a NumPy array into an existing [`Array`] of the same extents.
pub fn from_numpy_into<T: Scalar + Element, const N: usize>(
    value: &Bound<'_, PyAny>,
    out: &mut Array<T, N>,
) -> PyResult<()> {
    let arr = to_contiguous::<T>(value)?;
    let readonly = arr.extract::<PyReadonlyArrayDyn<'_, T>>()?;
    let src = readonly
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("from_numpy: non-contiguous value: {e}")))?;

    let dst = out.data_mut();
    if src.len() != dst.len() {
        return Err(PyValueError::new_err(format!(
            "from_numpy: expected {} elements, got {}",
            dst.len(),
            src.len(),
        )));
    }
    dst.clone_from_slice(src);

    Ok(())
}

/// Normalizes a Python value into a C-contiguous `T`-typed NumPy array via
/// `numpy.ascontiguousarray(value, dtype)`.
fn to_contiguous<'py, T: Element>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    py.import("numpy")?
        .getattr("ascontiguousarray")?
        .call1((value, numpy::dtype::<T>(py)))
}
