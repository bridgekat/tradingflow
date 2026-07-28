//! [`NativeArrayView`] / [`NativeArrayViewBool`] — Python-visible views over
//! a cell's `Array<T, N>`.
//!
//! PyO3 pyclasses cannot be generic, so the whole implementation lives in the
//! generic core [`RawArrayView<T>`] and each exposed element type gets a thin
//! `#[pyclass]` wrapper delegating to it: [`NativeArrayView`] (`f64`, the
//! data currency) and [`NativeArrayViewBool`] (`bool`, the clock signal
//! currency). Both speak the same protocol: `value()` / `to_numpy()` copy
//! out, `__array__` joins the NumPy protocol, `shape` reports extents, and
//! `__bool__` gives single-element views plain truthiness (`if clock:`),
//! mirroring NumPy's ambiguity error otherwise.
//!
//! Views are **read-only**: an operator's outputs are the array it *returns*
//! from `compute` (mirroring [`Segment::compute`](crate::graph::Segment),
//! which returns its output views), so there is no writable-view path.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{Element, PyArrayDyn};

use crate::data::{ArrayView, Scalar};

// [array-view-refactor] The const-rank refactor replaces the dynamic-rank
// `*mut Array<f64>` with a rank-erased pointer + an inline extents vector.
// Each per-leaf `PyArgs` impl knows its concrete rank `N` at the call site and
// reads the cell's static `[usize; N]` extents there; the pyclass stores them
// as a `Vec<usize>` so a single non-generic pyclass serves every rank (PyO3
// `#[pyclass]` cannot itself be const-generic). The flat buffer and a raw
// length are enough for the copy-out, since `Array<T, N>` is row-major
// contiguous — `data()` over the whole buffer is rank-independent.

/// Single-element truthiness per element type (NumPy semantics).
trait Truthy: Copy {
    fn truthy(self) -> bool;
}

impl Truthy for f64 {
    fn truthy(self) -> bool {
        self != 0.0
    }
}

impl Truthy for bool {
    fn truthy(self) -> bool {
        self
    }
}

/// The generic core of a read-only view over an input payload's data.
/// `value()` copies out. Valid only during the call that created it.
struct RawArrayView<T> {
    /// Pointer to the flat row-major buffer.
    data: *const T,
    /// Number of scalars (product of extents).
    len: usize,
    /// Element extents (the edge's static `[usize; N]`, as a runtime vector).
    extents: Vec<usize>,
    /// Owned row-major materialization of a strided input view (see
    /// [`bind_view`](Self::bind_view)); `data` points into it when present.
    /// Never read back — it exists to keep the pointee alive for the call.
    _backing: Option<Vec<T>>,
}

// SAFETY: single-threaded per compute, never retained/shared (module docs);
// the elements themselves are plain data.
unsafe impl<T: Send> Send for RawArrayView<T> {}
unsafe impl<T: Sync> Sync for RawArrayView<T> {}

impl<T: Element + Truthy> RawArrayView<T> {
    /// Copy the payload into a fresh NumPy array of the edge's shape.
    fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<T>> {
        let src = unsafe { std::slice::from_raw_parts(self.data, self.len) };
        let nd = ArrayD::from_shape_vec(IxDyn(&self.extents), src.to_vec())
            .expect("array shape/len mismatch");
        PyArrayDyn::from_owned_array(py, nd)
    }

    /// NumPy array protocol body, so `np.asarray(view)` / ufuncs work.
    fn array<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.value(py).into_any();
        match dtype {
            None => Ok(arr),
            Some(dt) => arr.call_method1("astype", (dt,)),
        }
    }

    /// Plain truthiness for a single-element view (`if clock:`); ambiguous
    /// for larger arrays, mirroring NumPy.
    fn truthiness(&self) -> PyResult<bool> {
        match self.len {
            1 => Ok(unsafe { *self.data }.truthy()),
            n => Err(PyValueError::new_err(format!(
                "truth value of a {n}-element view is ambiguous; use value()"
            ))),
        }
    }

    /// Bind a read-only view over an input payload (a possibly
    /// strided/broadcast [`ArrayView`]). A contiguous view is pointed at
    /// directly (zero-copy until Python reads it); a strided one is
    /// materialized row-major into an owned backing the view carries for the
    /// call.
    fn bind_view<const N: usize>(v: ArrayView<'_, T, N>) -> Self
    where
        T: Scalar,
    {
        let extents = v.extents().to_vec();
        let (data, len, backing) = match v.to_contiguous() {
            std::borrow::Cow::Borrowed(s) => (s.as_ptr(), s.len(), None),
            std::borrow::Cow::Owned(vec) => (vec.as_ptr(), vec.len(), Some(vec)),
        };
        Self {
            data,
            len,
            extents,
            _backing: backing,
        }
    }
}

/// Stamp out a thin `#[pyclass]` wrapper delegating to [`RawArrayView<$t>`].
macro_rules! native_array_view {
    ($(#[$doc:meta])* $name:ident, $t:ty) => {
        $(#[$doc])*
        #[pyclass]
        pub struct $name(RawArrayView<$t>);

        #[pymethods]
        impl $name {
            /// Copy the payload into a fresh NumPy array of the edge's shape.
            fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<$t>> {
                self.0.value(py)
            }

            /// Alias for [`value`](Self::value).
            fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<$t>> {
                self.0.value(py)
            }

            /// NumPy array protocol, so `np.asarray(view)` / ufuncs work.
            #[pyo3(signature = (dtype=None, copy=None))]
            fn __array__<'py>(
                &self,
                py: Python<'py>,
                dtype: Option<Bound<'py, PyAny>>,
                copy: Option<Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let _ = copy;
                self.0.array(py, dtype)
            }

            /// Plain truthiness for a single-element view (`if clock:`).
            fn __bool__(&self) -> PyResult<bool> {
                self.0.truthiness()
            }

            /// Element shape of the edge's payload.
            #[getter]
            fn shape(&self) -> Vec<usize> {
                self.0.extents.clone()
            }
        }

        impl $name {
            /// See [`RawArrayView::bind_view`].
            pub(super) fn bind_view<'py, const N: usize>(
                py: Python<'py>,
                v: ArrayView<'_, $t, N>,
            ) -> PyResult<Bound<'py, PyAny>> {
                Ok(Bound::new(py, Self(RawArrayView::bind_view(v)))?.into_any())
            }
        }
    };
}

native_array_view!(
    /// Read-only view over an `f64` array edge's payload. `value()` copies
    /// out into a fresh NumPy array; valid only during the call that created
    /// it.
    NativeArrayView,
    f64
);

native_array_view!(
    /// The `bool` sibling of [`NativeArrayView`] — the clock signal currency
    /// (`ClockArrayPort<N>` payloads). A rank-0 clock answers `if clock:`
    /// directly via `__bool__`.
    NativeArrayViewBool,
    bool
);
