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
        let stride = shape.iter().product::<usize>().max(1);
        debug_assert_eq!(initial.len(), stride);
        let vals = alloc_buf::<T>(stride);
        // SAFETY: `vals` points to `stride` uninitialised `T`s; we copy
        // `stride` elements from `initial`.
        unsafe {
            std::ptr::copy_nonoverlapping(initial.as_ptr(), vals, stride);
        }
        Self { stride, vals }
    }

    /// Create an observable with an **uninitialised** value buffer.
    ///
    /// The buffer must be written to (via [`write`] or [`vals_mut`]) before
    /// reading.  Used for operator output nodes whose initial values are
    /// computed during graph initialisation.
    pub fn new_uninit(shape: &[usize]) -> Self {
        let stride = shape.iter().product::<usize>().max(1);
        let vals = alloc_buf::<T>(stride);
        Self { stride, vals }
    }

    // -- Accessors ----------------------------------------------------------

    /// Number of values per element (product of shape dimensions, min 1).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Read the current value as a slice of length `stride`.
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

    // -- Mutation ------------------------------------------------------------

    /// Mutable access to the value buffer for in-place writes.
    ///
    /// Operators write their output directly into this buffer.
    #[inline(always)]
    pub fn vals_mut(&mut self) -> &mut [T] {
        // SAFETY: `vals[0..stride]` is always initialised and we have `&mut`.
        unsafe { std::slice::from_raw_parts_mut(self.vals, self.stride) }
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
        obs.write(&[0.0]); // must init before reading
        assert_eq!(obs.last(), &[0.0]);
    }

    #[test]
    fn new_with_initial() {
        let obs = Observable::<f64>::new(&[3], &[1.0, 2.0, 3.0]);
        assert_eq!(obs.stride(), 3);
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
        assert_eq!(obs.last(), &[1, 2, 3, 4, 5, 6]);
        obs.write(&[10, 20, 30, 40, 50, 60]);
        assert_eq!(obs.last(), &[10, 20, 30, 40, 50, 60]);
    }
}
