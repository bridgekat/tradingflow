//! Observable value — stores only the latest value of a time series node.
//!
//! An [`Observable<T>`] holds a fixed-size buffer of `stride` elements
//! (product of the shape dimensions) that is allocated once and overwritten on
//! each update.  It never grows.
//!
//! Timestamps are *not* stored — the scenario passes them externally.
//! There is no `has_value` flag — after graph initialisation all observables
//! hold valid values.  Missing data is represented via sentinel values
//! (NaN for floats, etc.).  Observables may be created uninitialised
//! (for operator outputs) and are set during the graph initialisation step.

use std::marker::PhantomData;

use ndarray::{ArrayView1, ArrayViewD, IxDyn};

// ---------------------------------------------------------------------------
// Observable<T>
// ---------------------------------------------------------------------------

/// Fixed-size value buffer for a single observable.
///
/// # Safety
///
/// All raw-pointer operations are behind safe methods.  The buffer is
/// allocated once in `new` / `new_uninit` and freed in `Drop`.
pub struct Observable<T: Copy> {
    shape: Box<[usize]>,
    stride: usize,
    vals: *mut T,
}

// SAFETY: Observable owns its allocation exclusively; no interior sharing.
unsafe impl<T: Copy + Send> Send for Observable<T> {}
// SAFETY: &Observable only exposes shared reads of the owned buffer.
unsafe impl<T: Copy + Sync> Sync for Observable<T> {}

impl<T: Copy> Observable<T> {
    /// Create an observable with an explicit initial value.
    ///
    /// `initial.len()` must equal the stride (product of shape dimensions,
    /// min 1).
    pub fn new(shape: &[usize], initial: &[T]) -> Self {
        let stride = shape.iter().product::<usize>();
        debug_assert_eq!(initial.len(), stride);
        let vals = alloc_buf::<T>(stride);
        // SAFETY: `vals` points to `stride` uninitialised `T`s; we copy
        // `stride` elements from `initial`.
        unsafe {
            std::ptr::copy_nonoverlapping(initial.as_ptr(), vals, stride);
        }
        Self { shape: shape.into(), stride, vals }
    }

    /// Create an observable with an **uninitialised** value buffer.
    ///
    /// The buffer must be written to (via [`write`] or [`vals_mut`]) before
    /// reading.  Used for operator output nodes whose initial values are
    /// computed during graph initialisation.
    pub fn new_uninit(shape: &[usize]) -> Self {
        let stride = shape.iter().product::<usize>();
        let vals = alloc_buf::<T>(stride);
        Self { shape: shape.into(), stride, vals }
    }

    // -- Accessors ----------------------------------------------------------

    /// Element shape (e.g. `&[2, 3]` for a 2x3 matrix element).
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of values per element (product of shape dimensions).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Read the current value as a flat slice of length `stride`.
    ///
    /// # Safety contract (logical)
    ///
    /// Caller must ensure the buffer has been initialised (via `new`,
    /// `write`, or `vals_mut`) before calling this.  After graph
    /// initialisation all observables are initialised.
    #[inline(always)]
    pub fn last(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.vals, self.stride) }
    }

    /// Zero-copy flat `ArrayView1` of the current value.
    #[inline]
    pub fn as_array_view_flat(&self) -> ArrayView1<'_, T> {
        unsafe { ArrayView1::from_shape_ptr(self.stride, self.vals as *const T) }
    }

    /// Zero-copy `ArrayViewD` with the element shape.
    ///
    /// - Scalar (shape `[]`): returns 0-dimensional view
    /// - Vector (shape `[3]`): returns shape `[3]`
    /// - Matrix (shape `[2, 3]`): returns shape `[2, 3]`
    #[inline]
    pub fn as_array_view(&self) -> ArrayViewD<'_, T> {
        unsafe { ArrayViewD::from_shape_ptr(IxDyn(&self.shape), self.vals as *const T) }
    }

    // -- Mutation ------------------------------------------------------------

    /// Mutable access to the value buffer for in-place writes.
    ///
    /// Operators write their output directly into this buffer.
    #[inline(always)]
    pub fn vals_mut(&mut self) -> &mut [T] {
        // SAFETY: `vals[0..stride]` is always initialised and we have `&mut`.
        unsafe { std::slice::from_raw_parts_mut(self.vals, self.stride) }
    }

    /// Raw byte pointer to the value buffer.
    ///
    /// The buffer contains `stride` values of type `T`, i.e.
    /// `stride * size_of::<T>()` bytes.
    #[inline(always)]
    pub fn vals_ptr(&mut self) -> *mut u8 {
        self.vals as *mut u8
    }

    /// Overwrite the value buffer from a slice.
    #[inline(always)]
    pub fn write(&mut self, value: &[T]) {
        debug_assert_eq!(value.len(), self.stride);
        // SAFETY: both slices have length `stride`.
        unsafe {
            std::ptr::copy_nonoverlapping(value.as_ptr(), self.vals, self.stride);
        }
    }
}

impl<T: Copy> Drop for Observable<T> {
    fn drop(&mut self) {
        dealloc_buf(self.vals, self.stride);
    }
}

// ---------------------------------------------------------------------------
// ObservableHandle<T>
// ---------------------------------------------------------------------------

/// Zero-cost typed handle into a [`Scenario`]'s node storage.
///
/// Carries the node index and a `PhantomData<T>` for compile-time type
/// checking at registration.  At runtime it is just a `usize`.
pub struct ObservableHandle<T: Copy> {
    pub(crate) index: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> ObservableHandle<T> {
    pub(crate) fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy> Clone for ObservableHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Copy> Copy for ObservableHandle<T> {}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

fn alloc_buf<T>(count: usize) -> *mut T {
    let count = count.max(1);
    let layout = std::alloc::Layout::array::<T>(count).expect("layout overflow");
    // SAFETY: layout is non-zero (count >= 1).
    let p = unsafe { std::alloc::alloc(layout) as *mut T };
    assert!(!p.is_null(), "allocation failed");
    p
}

fn dealloc_buf<T>(ptr: *mut T, count: usize) {
    let count = count.max(1);
    let layout = std::alloc::Layout::array::<T>(count).unwrap();
    // SAFETY: `ptr` was allocated with this layout.
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_uninit_scalar() {
        let mut obs = Observable::<f64>::new_uninit(&[]);
        assert_eq!(obs.stride(), 1);
        assert_eq!(obs.shape(), &[] as &[usize]);
        obs.write(&[0.0]); // must init before reading
        assert_eq!(obs.last(), &[0.0]);
    }

    #[test]
    fn new_with_initial() {
        let obs = Observable::<f64>::new(&[3], &[1.0, 2.0, 3.0]);
        assert_eq!(obs.stride(), 3);
        assert_eq!(obs.shape(), &[3]);
        assert_eq!(obs.last(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn write_overwrites() {
        let mut obs = Observable::<f64>::new_uninit(&[]);
        obs.write(&[42.0]);
        assert_eq!(obs.last(), &[42.0]);
        obs.write(&[99.0]);
        assert_eq!(obs.last(), &[99.0]);
    }

    #[test]
    fn vals_mut_in_place() {
        let mut obs = Observable::<f64>::new_uninit(&[2]);
        let buf = obs.vals_mut();
        buf[0] = 10.0;
        buf[1] = 20.0;
        assert_eq!(obs.last(), &[10.0, 20.0]);
    }

    #[test]
    fn strided_observable() {
        let mut obs = Observable::<i32>::new(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(obs.stride(), 6);
        assert_eq!(obs.shape(), &[2, 3]);
        assert_eq!(obs.last(), &[1, 2, 3, 4, 5, 6]);
        obs.write(&[10, 20, 30, 40, 50, 60]);
        assert_eq!(obs.last(), &[10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn array_view_scalar_zero_copy() {
        let obs = Observable::<f64>::new(&[], &[42.0]);
        let view = obs.as_array_view();
        // Scalar → 0-dimensional view.
        assert_eq!(view.shape(), &[] as &[usize]);
        assert_eq!(view.ndim(), 0);
        assert_eq!(view[[]], 42.0);
        // Verify zero-copy.
        assert_eq!(
            view.as_slice().unwrap().as_ptr(),
            obs.last().as_ptr(),
        );
    }

    #[test]
    fn array_view_vector_zero_copy() {
        let obs = Observable::<f64>::new(&[3], &[1.0, 2.0, 3.0]);
        let view = obs.as_array_view();
        assert_eq!(view.shape(), &[3]);
        assert_eq!(view[[0]], 1.0);
        assert_eq!(view[[2]], 3.0);
        assert_eq!(
            view.as_slice().unwrap().as_ptr(),
            obs.last().as_ptr(),
        );
    }

    #[test]
    fn array_view_matrix_zero_copy() {
        let obs = Observable::<f64>::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let view = obs.as_array_view();
        assert_eq!(view.shape(), &[2, 3]);
        // Row 0, col 2
        assert_eq!(view[[0, 2]], 3.0);
        // Row 1, col 0
        assert_eq!(view[[1, 0]], 4.0);
        assert_eq!(
            view.as_slice().unwrap().as_ptr(),
            obs.last().as_ptr(),
        );
    }

    #[test]
    fn flat_view_always_1d() {
        let obs = Observable::<f64>::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let flat = obs.as_array_view_flat();
        assert_eq!(flat.len(), 6);
        assert_eq!(flat[3], 4.0);
    }
}
