//! [`NativeArrayView`] — Python-visible view over a cell's `Array<f64, N>`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};

use crate::data::{Array, ArrayView, Layout};

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
    pub(super) fn bind<'py, const N: usize>(
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
    pub(super) fn bind_view<'py, const N: usize>(
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
