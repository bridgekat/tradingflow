//! [`NativeView`] — a read-only PEP 3118 buffer over Rust-owned memory.
//!
//! One [`NativeView`] is one buffer: a pointer into a graph payload, plus the
//! shape/stride metadata needed to describe it, plus the host control
//! [`invalidate`](NativeView::invalidate) that make the borrow contract of
//! [the module docs](super) enforceable.
//!
//! # Why the buffer protocol
//!
//! The alternative, `__array_interface__`, is a *passive* dict: NumPy copies
//! the raw address out of it and the exporter never learns that it happened.
//! PEP 3118 is *active* — acquisition runs `__getbuffer__` and disposal runs
//! `__releasebuffer__` — which buys exactly the two things the contract needs:
//!
//! - **Exact accounting.** Every zero-copy consumer (an `ndarray` built over
//!   the view, a `memoryview`, anything downstream of them) holds a live export
//!   for as long as it can dereference the pointer. So
//!   [`exports`](NativeView::exports) is not a heuristic: at the end of a call
//!   it is the precise number of outstanding raw-pointer captures.
//! - **Revocation of future acquisitions.** A retained view object is not
//!   itself dangerous — only a captured pointer is. After
//!   [`invalidate`](NativeView::invalidate), a later `np.asarray(view)` runs
//!   `__getbuffer__` and raises `BufferError` instead of reading freed memory.
//!
//! Neither is possible with the array interface, which is why this type
//! deliberately implements **only** the buffer protocol.
//!
//! # Rank erasure
//!
//! `#[pyclass]` cannot be const-generic, so the rank `N` is erased here into
//! runtime `shape`/`strides` vectors. The per-leaf binding site knows the
//! concrete `N` and reads the static `[usize; N]` extents there, so a single
//! non-generic class serves every rank — including rank 0 (`ndim == 0`, which
//! NumPy and `memoryview` both accept).
//!
//! # Series are two buffers
//!
//! PEP 3118 exports one buffer per object, so a [`SeriesView`] binds as a
//! *pair*: [`series_values`](NativeView::series_values), a rank-`N + 1` buffer
//! whose axis 0 is time, and [`series_instants`](NativeView::series_instants),
//! an `int64` buffer of nanosecond timestamps. Note the rank-erased
//! representation is what makes the `N + 1` values buffer expressible at all —
//! stable Rust cannot spell that type.
//!
//! Timestamps travel as `int64` because PEP 3118 has no `datetime64` format
//! code. [`Instant`] is `repr(transparent)` over `i64` nanoseconds and NumPy's
//! `datetime64[ns]` has the same layout, so the Python wrapper recovers the
//! typed form with a zero-copy `np.asarray(ts).view("datetime64[ns]")`.

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::atomic::{self, AtomicPtr, AtomicUsize, Ordering};

use super::{DType, NativeScalar};
use crate::data::{ArrayView, Instant, Layout, SeriesView};

/// A read-only, strided buffer over a graph payload, exported to Python via
/// PEP 3118.
///
/// The pointee is Rust-owned memory borrowed for the duration of one Python
/// call (see [the module docs](super) for the ownership model). Python code
/// consumes it zero-copy — `np.asarray(view)` yields a read-only array sharing
/// the graph's memory — and must not keep it, or anything derived from it,
/// beyond the call. The host enforces that with [`exports`](Self::exports).
///
/// This class is deliberately minimal: it carries a buffer and its metadata,
/// nothing else. The typed, documented wrappers Python operators actually see
/// live in `python/tradingflow/views.py`.
#[pyclass(frozen)]
pub struct NativeView {
    /// The first addressed byte. Can be reset to null by [`Self::invalidate`].
    /// Can never change again once reset.
    data: AtomicPtr<()>,
    /// Element type of the buffer.
    dtype: DType,
    /// Per-axis extents. Rank-erased: `shape.len()` is the rank.
    shape: Vec<ffi::Py_ssize_t>,
    /// Per-axis strides **in bytes** (PEP 3118 counts bytes; the data layer
    /// counts elements).
    strides: Vec<ffi::Py_ssize_t>,
    /// Number of elements — the product of `shape`, so `1` at rank 0.
    count: usize,
    /// Outstanding buffer exports: incremented by `__getbuffer__`, decremented
    /// by `__releasebuffer__`. Nonzero after a call means a raw pointer into
    /// the graph escaped it.
    exports: AtomicUsize,
}

impl NativeView {
    /// Binds an [`ArrayView`] as a rank-`N` buffer.
    ///
    /// Zero-copy including strides: a sliced or transposed view is exported as
    /// it lies, with no materialization, since PEP 3118 carries strides.
    ///
    /// # Safety
    ///
    /// The returned view holds a raw pointer into `v`'s buffer with the borrow
    /// erased, so nothing left in the type system tracks when that memory dies.
    /// The caller must
    ///
    /// - call [`invalidate`](Self::invalidate) before `v`'s memory is freed,
    ///   moved or written to, and
    /// - treat an `Err` from that call as fatal — an export outliving the
    ///   borrow cannot be revoked, so its captured pointer is already doomed.
    ///
    /// [`Scope`](super::Scope) discharges both obligations for a whole call's
    /// worth of views; prefer it unless you are hosting a segment by hand.
    pub unsafe fn array<'py, T: NativeScalar, const N: usize>(
        py: Python<'py>,
        v: ArrayView<'_, T, N>,
    ) -> PyResult<Bound<'py, NativeView>> {
        let layout = v.layout();
        let data = v.data();
        debug_assert!(data.len() >= layout.span());
        Self::bind(
            py,
            T::DTYPE,
            data.as_ptr().cast(),
            layout.extents().iter().map(|&e| e as _).collect(),
            layout.strides().iter().map(|&s| s as _).collect(),
        )
    }

    /// Binds the timestamps of a [`SeriesView`] as an `int64` buffer of
    /// nanoseconds.
    ///
    /// PEP 3118 cannot express `datetime64`; [`Instant`] is `repr(transparent)`
    /// over `i64` and shares NumPy's `datetime64[ns]` layout, so the Python
    /// wrapper recovers the typed form zero-copy with
    /// `np.asarray(ts).view("datetime64[ns]")`.
    ///
    /// # Safety
    ///
    /// As [`array`](Self::array), over `s`'s timestamps.
    pub unsafe fn series_instants<'py, T: NativeScalar, const N: usize>(
        py: Python<'py>,
        s: SeriesView<'_, T, N>,
    ) -> PyResult<Bound<'py, NativeView>> {
        let nanos = Instant::as_nanos_slice(s.instants());
        Self::bind(
            py,
            DType::Int64,
            nanos.as_ptr().cast(),
            vec![nanos.len() as ffi::Py_ssize_t],
            vec![1],
        )
    }

    /// Binds the values of a [`SeriesView`] as a rank-`N + 1` buffer whose
    /// axis 0 is time.
    ///
    /// Pairs with [`series_instants`](Self::series_instants), which carries the
    /// timestamps of the same window.
    ///
    /// # Safety
    ///
    /// As [`array`](Self::array), over `s`'s values.
    pub unsafe fn series_values<'py, T: NativeScalar, const N: usize>(
        py: Python<'py>,
        s: SeriesView<'_, T, N>,
    ) -> PyResult<Bound<'py, NativeView>> {
        let layout = s.layout();
        let data = s.data();
        debug_assert!(data.len() >= layout.span_ext(s.len(), s.stride()));

        // Prepend the time axis: one element every `stride` scalars.
        let mut shape = Vec::with_capacity(N + 1);
        let mut strides = Vec::with_capacity(N + 1);
        shape.push(s.len() as ffi::Py_ssize_t);
        strides.push(s.stride() as ffi::Py_ssize_t);
        shape.extend(layout.extents().iter().map(|&e| e as ffi::Py_ssize_t));
        strides.extend(layout.strides().iter().map(|&s| s as ffi::Py_ssize_t));

        Self::bind(py, T::DTYPE, data.as_ptr().cast(), shape, strides)
    }

    /// Closes the borrow window, returning [`Ok`] if there are no outstanding
    /// exports, or [`Err`] with the number of outstanding exports.
    ///
    /// Later acquisitions raise `BufferError` rather than reading a dangling
    /// pointer.
    pub fn invalidate(&self) -> Result<(), usize> {
        self.data.store(ptr::null_mut(), Ordering::Relaxed);
        atomic::fence(Ordering::SeqCst);
        let exports = self.exports.load(Ordering::Relaxed);
        if exports == 0 {
            atomic::fence(Ordering::Acquire);
            return Ok(());
        }
        Err(exports)
    }

    /// A short `dtype[extents]` label for diagnostics.
    pub fn describe(&self) -> String {
        format!("{}{:?}", self.dtype.name(), self.shape)
    }

    /// The single construction path. `strides` arrives counted in **elements**,
    /// as the data layer counts them, and is converted to the bytes PEP 3118
    /// wants here — so the conversion lives in exactly one place.
    fn bind<'py>(
        py: Python<'py>,
        dtype: DType,
        data: *const (),
        shape: Vec<ffi::Py_ssize_t>,
        mut strides: Vec<ffi::Py_ssize_t>,
    ) -> PyResult<Bound<'py, NativeView>> {
        debug_assert_eq!(shape.len(), strides.len());
        let count = shape.iter().copied().product::<ffi::Py_ssize_t>() as usize;
        for stride in &mut strides {
            *stride *= dtype.itemsize() as ffi::Py_ssize_t;
        }
        Bound::new(
            py,
            NativeView {
                data: AtomicPtr::new(data.cast_mut()),
                dtype,
                shape,
                strides,
                count,
                exports: AtomicUsize::new(0),
            },
        )
    }

    /// Row-major contiguity, computed exactly as CPython's
    /// `PyBuffer_IsContiguous` does — extent-1 axes carry no constraint, and an
    /// empty buffer is contiguous — so our negotiation agrees with the
    /// interpreter's own checks.
    fn is_c_contiguous(&self) -> bool {
        if self.count == 0 {
            return true;
        }
        let mut expected = self.dtype.itemsize() as ffi::Py_ssize_t;
        for i in (0..self.shape.len()).rev() {
            let extent = self.shape[i];
            if extent > 1 && self.strides[i] != expected {
                return false;
            }
            expected *= extent;
        }
        true
    }

    /// Column-major contiguity; the mirror of [`is_c_contiguous`](Self::is_c_contiguous).
    fn is_f_contiguous(&self) -> bool {
        if self.count == 0 {
            return true;
        }
        let mut expected = self.dtype.itemsize() as ffi::Py_ssize_t;
        for i in 0..self.shape.len() {
            let extent = self.shape[i];
            if extent > 1 && self.strides[i] != expected {
                return false;
            }
            expected *= extent;
        }
        true
    }
}

#[pymethods]
impl NativeView {
    /// Number of buffer exports currently outstanding. For internal use only.
    #[getter]
    pub fn exports(&self) -> usize {
        self.exports.load(Ordering::Relaxed)
    }

    /// Whether the view still points at live memory. For internal use only.
    #[getter]
    pub fn valid(&self) -> bool {
        !self.data.load(Ordering::Relaxed).is_null()
    }

    /// Per-axis extents. Empty at rank 0.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.iter().map(|&e| e as usize).collect()
    }

    /// Per-axis strides, in bytes.
    #[getter]
    fn strides(&self) -> Vec<isize> {
        self.strides.clone()
    }

    /// The rank of the buffer.
    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Size of one element in bytes.
    #[getter]
    fn itemsize(&self) -> usize {
        self.dtype.itemsize()
    }

    /// The PEP 3118 format code (`"d"`, `"q"`, `"?"`, …).
    #[getter]
    fn format(&self) -> &'static str {
        self.dtype
            .format()
            .to_str()
            .expect("dtype format codes are ASCII")
    }

    /// The NumPy dtype name (`"float64"`, `"int64"`, `"bool"`, …).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "<NativeView {} exports={} valid={}>",
            self.describe(),
            self.exports(),
            self.valid(),
        )
    }

    /// Export the buffer (PEP 3118).
    ///
    /// Read-only always: `PyBUF_WRITABLE` is refused, so no pointer with write
    /// provenance reaches Python and an operator's outputs can only be the
    /// array it returns.
    ///
    /// # Safety
    ///
    /// Called by the interpreter with a valid `Py_buffer` slot; `shape` and
    /// `strides` are handed out as pointers into `self`, which stays alive
    /// because `view.obj` holds a strong reference to it for the export's
    /// lifetime.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("Py_buffer slot is null"));
        }

        // CPython expects a failed acquisition to leave `obj` null; clear it up
        // front so every early return below is well-formed.
        unsafe { (*view).obj = ptr::null_mut() };

        let this = slf.get();
        let wants = |flag: c_int| flags & flag == flag;

        // A consumer that did not ask for strides can only be handed a
        // C-contiguous buffer, since it will read the block flat.
        if !wants(ffi::PyBUF_STRIDES) && !this.is_c_contiguous() {
            return Err(PyBufferError::new_err(format!(
                "native view {} is strided; the request must include \
                 PyBUF_STRIDES",
                this.describe(),
            )));
        }
        if wants(ffi::PyBUF_C_CONTIGUOUS) && !this.is_c_contiguous() {
            return Err(PyBufferError::new_err(format!(
                "native view {} is not C-contiguous",
                this.describe(),
            )));
        }
        if wants(ffi::PyBUF_F_CONTIGUOUS) && !this.is_f_contiguous() {
            return Err(PyBufferError::new_err(format!(
                "native view {} is not Fortran-contiguous",
                this.describe(),
            )));
        }
        if wants(ffi::PyBUF_ANY_CONTIGUOUS) && !(this.is_c_contiguous() || this.is_f_contiguous()) {
            return Err(PyBufferError::new_err(format!(
                "native view {} is not contiguous",
                this.describe(),
            )));
        }
        if wants(ffi::PyBUF_WRITABLE) {
            return Err(PyBufferError::new_err(format!(
                "native view {} is read-only",
                this.describe(),
            )));
        }

        // Publish the export before reading the data pointer.
        this.exports.fetch_add(1, Ordering::Relaxed);
        atomic::fence(Ordering::SeqCst);
        let data = this.data.load(Ordering::Relaxed);
        if data.is_null() {
            atomic::fence(Ordering::Release);
            this.exports.fetch_sub(1, Ordering::Relaxed);
            return Err(PyBufferError::new_err(format!(
                "native view {} has expired: its memory was only borrowed for \
                 the duration of the call that produced it",
                this.describe(),
            )));
        }

        let itemsize = this.dtype.itemsize();
        unsafe {
            (*view).buf = data as *mut c_void;
            (*view).len = (this.count * itemsize) as ffi::Py_ssize_t;
            (*view).itemsize = itemsize as ffi::Py_ssize_t;
            (*view).readonly = 1;
            (*view).format = if wants(ffi::PyBUF_FORMAT) {
                this.dtype.format().as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            };
            // `ndim` stays the true rank even when `shape` is withheld, which
            // is what NumPy's own exporter does.
            (*view).ndim = this.shape.len() as c_int;
            (*view).shape = if wants(ffi::PyBUF_ND) {
                this.shape.as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            };
            (*view).strides = if wants(ffi::PyBUF_STRIDES) {
                this.strides.as_ptr().cast_mut()
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
            // Keep the view alive for as long as the consumer holds the
            // buffer: this reference is what makes the `shape`/`strides`
            // pointers above sound.
            (*view).obj = slf.clone().into_ptr();
        }
        Ok(())
    }

    /// Release an export (PEP 3118).
    ///
    /// # Safety
    ///
    /// Called by the interpreter, once per successful `__getbuffer__`. The
    /// strong reference in `view.obj` is dropped by the interpreter afterwards.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        atomic::fence(Ordering::Release);
        if self.exports.fetch_sub(1, Ordering::Relaxed) == 0 {
            panic!("unbalanced __releasebuffer__");
        }
    }
}
