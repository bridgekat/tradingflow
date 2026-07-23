//! [`NativeSeriesView`] — Python-visible view over a cell's `Series<f64>`
//! (recorded history).

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn};

use crate::data::{Instant, Layout, SeriesView};

/// Read-only view over a recorded-history window (a [`SeriesView`] carried by
/// a `SeriesPort` edge): positional history access with **logical** indices —
/// index `base` is the oldest retained row, `len` counts every row ever
/// recorded, and an index is stable across the record's trims. Valid only
/// during the call that created it. Rank-erased like
/// [`NativeArrayView`](super::NativeArrayView): the per-leaf
/// [`PyArgs`](super::PyArgs) impl knows the concrete rank `N` at the bind site
/// and captures the window's raw parts there, so a single non-generic pyclass
/// serves every rank.
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
        self.logical_len()
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
        let (start, end) = self.clamp_range(start, end);
        // SAFETY: the cell outlives this call and [start, end) ⊆ [base, n).
        let flat = unsafe { self.window(start, end - start) };
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
        self.at(py, self.logical_len() as isize - 1)
    }

    /// Element at logical index `i` (supports negative indexing). Raises
    /// `IndexError` if the index has been dropped by the retention bound.
    fn at<'py>(&self, py: Python<'py>, i: isize) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let n = self.logical_len() as isize;
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
        let elem = unsafe { self.window(idx, 1) };
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
        let (start, end) = self.clamp_range(start, end);
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
        let ind = sl.indices(self.logical_len() as isize)?;
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
    /// rank-erased. The view's logical frame carries through: Python sees the
    /// same trim-stable indices as native consumers.
    pub(super) fn bind<'py, const N: usize>(
        py: Python<'py>,
        s: SeriesView<'_, f64, N>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Python reads the buffer flat, at `(i - base) * stride + j`, so the
        // window must be packed row-major — which is exactly when a series
        // view yields one slice.
        let values = s.as_slice().ok_or_else(|| {
            PyValueError::new_err("bind: series window is not contiguous row-major")
        })?;
        let view = NativeSeriesView {
            values: values.as_ptr(),
            timestamps: s.instants().as_ptr(),
            retained: s.len(),
            base: s.range().start,
            stride: s.layout().len(),
            extents: s.extents().to_vec(),
        };
        Ok(Bound::new(py, view)?.into_any())
    }

    /// Logical length: evicted elements plus retained ones.
    fn logical_len(&self) -> usize {
        self.base + self.retained
    }

    /// Clamp a logical `[start, end)` request onto the retained window
    /// `[base, len)`; an open `end` runs to the logical end.
    fn clamp_range(&self, start: usize, end: Option<usize>) -> (usize, usize) {
        let n = self.logical_len();
        let start = start.clamp(self.base, n);
        (start, end.unwrap_or(n).clamp(start, n))
    }

    /// The `count` elements from logical index `start`, as flat row-major
    /// scalars.
    ///
    /// # Safety
    ///
    /// `[start, start + count)` must lie inside the retained window
    /// `[base, base + retained)`, which must still be alive (see module docs).
    unsafe fn window(&self, start: usize, count: usize) -> &[f64] {
        unsafe {
            std::slice::from_raw_parts(
                self.values.add((start - self.base) * self.stride),
                count * self.stride,
            )
        }
    }
}
