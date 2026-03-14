//! Append-only time series with `i64` nanosecond timestamps and flat `T` values.
//!
//! Timestamps and values share a single `len`/`cap` pair — they always grow
//! together.  Values are stored as `cap * stride` contiguous `T`s; element *i*
//! occupies `values[i*stride .. (i+1)*stride]`.
//!
//! Memory is managed manually (alloc / realloc) so that the hot-path
//! [`append_unchecked`] compiles to a capacity check + two pointer writes.

use std::marker::PhantomData;

const INITIAL_CAPACITY: usize = 16;

// ---------------------------------------------------------------------------
// Series<T>
// ---------------------------------------------------------------------------

/// Append-only time series.
///
/// # Safety
///
/// All raw-pointer operations are encapsulated behind safe methods.  The
/// invariant `len <= cap` is maintained by every mutating method and `Drop`
/// deallocates exactly what was allocated.
pub struct Series<T: Copy> {
    stride: usize,
    len: usize,
    cap: usize,
    ts: *mut i64,
    vals: *mut T,
}

// SAFETY: Series owns its allocations exclusively; no interior sharing.
unsafe impl<T: Copy + Send> Send for Series<T> {}
// SAFETY: &Series only exposes shared reads of the owned buffers.
unsafe impl<T: Copy + Sync> Sync for Series<T> {}

impl<T: Copy> Series<T> {
    /// Create a new series with shape-derived stride and default capacity.
    pub fn new(shape: &[usize]) -> Self {
        Self::with_capacity(shape, INITIAL_CAPACITY)
    }

    /// Create a new series with shape-derived stride and at least `cap` slots.
    pub fn with_capacity(shape: &[usize], cap: usize) -> Self {
        let stride = shape.iter().product::<usize>().max(1);
        let cap = cap.max(1);
        Self {
            stride,
            len: 0,
            cap,
            ts: alloc_buf::<i64>(cap),
            vals: alloc_buf::<T>(cap * stride),
        }
    }

    // -- Accessors ----------------------------------------------------------

    /// Number of elements currently stored.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Number of values per element (product of shape dimensions, min 1).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Value slice for the most recently appended element.
    ///
    /// # Panics
    ///
    /// Panics if the series is empty.
    #[inline(always)]
    pub fn last(&self) -> &[T] {
        debug_assert!(self.len > 0, "last() on empty series");
        // SAFETY: `len > 0` guarantees the range is initialised.
        unsafe {
            std::slice::from_raw_parts(
                self.vals.add((self.len - 1) * self.stride),
                self.stride,
            )
        }
    }

    /// Timestamp of the most recently appended element.
    #[inline(always)]
    pub fn last_timestamp(&self) -> i64 {
        debug_assert!(self.len > 0, "last_timestamp() on empty series");
        unsafe { *self.ts.add(self.len - 1) }
    }

    /// View all timestamps as a slice.
    #[inline]
    pub fn timestamps(&self) -> &[i64] {
        // SAFETY: `ts[0..len]` is initialised.
        unsafe { std::slice::from_raw_parts(self.ts, self.len) }
    }

    /// View all values as a flat slice (length = `len * stride`).
    #[inline]
    pub fn values(&self) -> &[T] {
        // SAFETY: `vals[0..len*stride]` is initialised.
        unsafe { std::slice::from_raw_parts(self.vals, self.len * self.stride) }
    }

    /// Copy timestamps into a new `Vec`.
    pub fn timestamps_to_vec(&self) -> Vec<i64> {
        self.timestamps().to_vec()
    }

    /// Copy values into a new `Vec`.
    pub fn values_to_vec(&self) -> Vec<T> {
        self.values().to_vec()
    }

    // -- Mutation ------------------------------------------------------------

    /// Append a `(timestamp, value)` pair **without** checking monotonicity.
    ///
    /// # Safety contract (logical)
    ///
    /// Caller must ensure timestamps are appended in strictly increasing order.
    /// Violating this does not cause UB but breaks the Series ordering invariant.
    #[inline(always)]
    pub fn append_unchecked(&mut self, ts: i64, value: &[T]) {
        debug_assert_eq!(value.len(), self.stride);
        if self.len == self.cap {
            self.grow();
        }
        // SAFETY: `len < cap` after a possible grow; both buffers are large enough.
        unsafe {
            self.ts.add(self.len).write(ts);
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                self.vals.add(self.len * self.stride),
                self.stride,
            );
        }
        self.len += 1;
    }

    /// Reserve the next value slot without advancing `len`.
    ///
    /// Write into the returned slice, then call [`commit`] to publish the
    /// element.  If the operator decides not to produce output, simply do
    /// nothing — the slot is not yet visible.
    #[inline(always)]
    pub fn reserve_slot(&mut self) -> &mut [T] {
        if self.len == self.cap {
            self.grow();
        }
        // SAFETY: `len < cap`; the slice is within the allocation.
        unsafe {
            let p = self.vals.add(self.len * self.stride);
            std::slice::from_raw_parts_mut(p, self.stride)
        }
    }

    /// Publish the previously reserved slot with the given timestamp.
    ///
    /// Must be called exactly once after each successful [`reserve_slot`] write.
    #[inline(always)]
    pub fn commit(&mut self, ts: i64) {
        debug_assert!(self.len < self.cap, "commit without prior reserve_slot");
        // SAFETY: `len < cap` (reserve_slot guaranteed space).
        unsafe { self.ts.add(self.len).write(ts) };
        self.len += 1;
    }

    /// Remove all elements, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    // -- Internal -----------------------------------------------------------

    /// Double the capacity.  Marked `#[cold]` because it should rarely execute
    /// on the hot path (geometric growth).
    #[inline(never)]
    #[cold]
    fn grow(&mut self) {
        let new_cap = self.cap * 2;
        self.ts = realloc_buf(self.ts, self.cap, new_cap);
        self.vals = realloc_buf(self.vals, self.cap * self.stride, new_cap * self.stride);
        self.cap = new_cap;
    }
}

impl<T: Copy> Drop for Series<T> {
    fn drop(&mut self) {
        dealloc_buf(self.ts, self.cap);
        dealloc_buf(self.vals, self.cap * self.stride);
    }
}

// ---------------------------------------------------------------------------
// ErasedSeries
// ---------------------------------------------------------------------------

/// Type-erased wrapper around a heap-allocated `Series<T>`.
///
/// The actual `Series<T>` lives behind `ptr`; `drop_fn` knows the concrete
/// type and is called on drop.  This is the C-style `void*` pattern — all
/// type checking happens at registration time via [`SeriesHandle<T>`].
pub(crate) struct ErasedSeries {
    pub(crate) ptr: *mut u8,
    drop_fn: unsafe fn(*mut u8),
}

impl ErasedSeries {
    pub(crate) fn new<T: Copy>(shape: &[usize]) -> Self {
        Self {
            ptr: Box::into_raw(Box::new(Series::<T>::new(shape))) as *mut u8,
            drop_fn: drop_series::<T>,
        }
    }

    pub(crate) fn with_capacity<T: Copy>(shape: &[usize], cap: usize) -> Self {
        Self {
            ptr: Box::into_raw(Box::new(Series::<T>::with_capacity(shape, cap))) as *mut u8,
            drop_fn: drop_series::<T>,
        }
    }
}

impl Drop for ErasedSeries {
    fn drop(&mut self) {
        // SAFETY: `drop_fn` matches the concrete type that was used to create `ptr`.
        unsafe { (self.drop_fn)(self.ptr) }
    }
}

unsafe fn drop_series<T: Copy>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut Series<T>)) };
}

// ---------------------------------------------------------------------------
// SeriesHandle<T>
// ---------------------------------------------------------------------------

/// Zero-cost typed handle into a [`Scenario`]'s series storage.
///
/// Carries the series index and a `PhantomData<T>` for compile-time type
/// checking at registration.  At runtime it is just a `usize`.
pub struct SeriesHandle<T: Copy> {
    pub(crate) index: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> SeriesHandle<T> {
    pub(crate) fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy> Clone for SeriesHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Copy> Copy for SeriesHandle<T> {}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

fn alloc_buf<T>(count: usize) -> *mut T {
    let layout = std::alloc::Layout::array::<T>(count).expect("layout overflow");
    // SAFETY: layout is non-zero (count >= 1 enforced by callers).
    let p = unsafe { std::alloc::alloc(layout) as *mut T };
    assert!(!p.is_null(), "allocation failed");
    p
}

fn realloc_buf<T>(ptr: *mut T, old_count: usize, new_count: usize) -> *mut T {
    let old_layout = std::alloc::Layout::array::<T>(old_count).unwrap();
    let new_size = std::alloc::Layout::array::<T>(new_count).unwrap().size();
    // SAFETY: `ptr` was allocated with `old_layout`; `new_size >= old_size`.
    let p = unsafe { std::alloc::realloc(ptr as *mut u8, old_layout, new_size) as *mut T };
    assert!(!p.is_null(), "reallocation failed");
    p
}

fn dealloc_buf<T>(ptr: *mut T, count: usize) {
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
    fn append_and_read() {
        let mut s = Series::<f64>::new(&[]);
        s.append_unchecked(1, &[10.0]);
        s.append_unchecked(2, &[20.0]);
        s.append_unchecked(3, &[30.0]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.timestamps(), &[1, 2, 3]);
        assert_eq!(s.values(), &[10.0, 20.0, 30.0]);
        assert_eq!(s.last(), &[30.0]);
    }

    #[test]
    fn reserve_and_commit() {
        let mut s = Series::<f64>::new(&[]);
        let slot = s.reserve_slot();
        slot[0] = 42.0;
        s.commit(100);
        assert_eq!(s.len(), 1);
        assert_eq!(s.last(), &[42.0]);
        assert_eq!(s.last_timestamp(), 100);
    }

    #[test]
    fn strided_series() {
        let mut s = Series::<f64>::new(&[3]);
        s.append_unchecked(1, &[1.0, 2.0, 3.0]);
        s.append_unchecked(2, &[4.0, 5.0, 6.0]);
        assert_eq!(s.len(), 2);
        assert_eq!(s.stride(), 3);
        assert_eq!(s.last(), &[4.0, 5.0, 6.0]);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn grow_beyond_initial_capacity() {
        let mut s = Series::<i32>::with_capacity(&[], 2);
        for i in 0..100 {
            s.append_unchecked(i as i64, &[i]);
        }
        assert_eq!(s.len(), 100);
        assert_eq!(s.last(), &[99]);
    }

    #[test]
    fn clear_resets_length() {
        let mut s = Series::<f64>::new(&[]);
        s.append_unchecked(1, &[1.0]);
        s.clear();
        assert!(s.is_empty());
        // Can append again after clear.
        s.append_unchecked(2, &[2.0]);
        assert_eq!(s.len(), 1);
    }
}
