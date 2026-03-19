use ndarray::{ArrayView1, ArrayViewD, ArrayViewMut1, ArrayViewMutD, Ix1, IxDyn};

const INITIAL_CAPACITY: usize = 16;
const INITIAL_TIMESTAMP: i64 = i64::MIN;

/// A time series of *n*-dimensional array values, with 64-bit timestamps
/// representing nanoseconds since the UNIX epoch (1970-01-01 00:00:00 UTC).
///
/// A [`Series<T>`] holds a contiguous buffer of `stride * cap` scalar values
/// where `stride` is the product of the element shape dimensions and `cap` is the
/// current capacity. The series grows by doubling the capacity when needed.
/// Timestamps are stored in a separate buffer of length `cap`.
///
/// Every time series is created with an explicit initial value.
/// Missing data should be represented via sentinel values (0, NaN, etc.).
/// The initial timestamp is set to the minimum possible value ([`i64::MIN`]),
/// representing the state before any updates.
#[derive(Debug)]
pub struct Series<T: Copy> {
    values: *mut T,
    index: *mut i64,
    stride: usize,
    cap: usize,
    len_shape: Box<[usize]>,
}

impl<T: Copy> Series<T> {
    /// Create a new series with an explicit initial value and default capacity.
    ///
    /// `value.len()` must equal the stride (product of shape dimensions).
    pub fn new(shape: &[usize], value: &[T]) -> Self {
        let stride = shape.iter().product::<usize>();
        let cap = INITIAL_CAPACITY.max(1);
        let len_shape = std::iter::once(1)
            .chain(shape.iter().copied())
            .collect::<Box<[_]>>();
        let values = alloc_buf::<T>(stride * cap);
        let index = alloc_buf::<i64>(cap);
        assert!(value.len() == stride);
        // SAFETY: `index` have length at least 1.
        unsafe { index.write(INITIAL_TIMESTAMP) };
        // SAFETY: both slices have length at least `stride`, and `T: Copy`.
        unsafe { std::ptr::copy_nonoverlapping(value.as_ptr(), values, stride) };
        Self {
            values,
            index,
            stride,
            cap,
            len_shape,
        }
    }

    /// Whether the series is empty. Always returns false due to the initial value.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len_shape[0] == 0
    }

    /// Number of observations currently stored, including the initial value.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len_shape[0]
    }

    /// Element shape (e.g. `&[2, 3]` for a 2x3 matrix element).
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.len_shape[1..]
    }

    /// Number of values per element (product of shape dimensions).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The value buffer as a slice of length `len * stride`.
    #[inline(always)]
    pub fn values(&self) -> &[T] {
        // SAFETY: `values` has at least `stride * len` elements allocated.
        unsafe { std::slice::from_raw_parts(self.values, self.stride * self.len()) }
    }

    /// The value buffer as a mutable slice of length `stride * len`.
    #[inline(always)]
    pub fn values_mut(&mut self) -> &mut [T] {
        // SAFETY: `values` has at least `stride * len` elements allocated, and we have `&mut`.
        unsafe { std::slice::from_raw_parts_mut(self.values, self.stride * self.len()) }
    }

    /// Array view of the value buffer with shape `[len, ...element_shape]`.
    #[inline(always)]
    pub fn values_view(&self) -> ArrayViewD<'_, T> {
        // SAFETY: the shape is valid for the buffer which has length `stride * len`.
        unsafe { ArrayViewD::from_shape_ptr(IxDyn(&self.len_shape), self.values) }
    }

    /// Mutable array view of the value buffer with shape `[len, ...element_shape]`.
    #[inline(always)]
    pub fn values_view_mut(&mut self) -> ArrayViewMutD<'_, T> {
        // SAFETY: the shape is valid for the buffer which has length `stride * len`.
        unsafe { ArrayViewMutD::from_shape_ptr(IxDyn(&self.len_shape), self.values) }
    }

    /// The timestamp buffer as a slice of length `len`.
    #[inline(always)]
    pub fn index(&self) -> &[i64] {
        // SAFETY: `index` has at least `len` elements allocated.
        unsafe { std::slice::from_raw_parts(self.index, self.len()) }
    }

    /// The timestamp buffer as a mutable slice of length `len`.
    #[inline(always)]
    pub fn index_mut(&mut self) -> &mut [i64] {
        // SAFETY: `index` has at least `len` elements allocated, and we have `&mut`.
        unsafe { std::slice::from_raw_parts_mut(self.index, self.len()) }
    }

    /// Array view of the timestamp buffer with shape `[len]`.
    #[inline(always)]
    pub fn index_view(&self) -> ArrayView1<'_, i64> {
        // SAFETY: the shape is valid for the buffer which has length `len`.
        unsafe { ArrayView1::from_shape_ptr(Ix1(self.len()), self.index) }
    }

    /// Mutable array view of the timestamp buffer with shape `[len]`.
    #[inline(always)]
    pub fn index_view_mut(&mut self) -> ArrayViewMut1<'_, i64> {
        // SAFETY: the shape is valid for the buffer which has length `len`, and we have `&mut`.
        unsafe { ArrayViewMut1::from_shape_ptr(Ix1(self.len()), self.index) }
    }

    /// The i-th element as a slice of length `stride`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `i < self.len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> &[T] {
        // SAFETY: `values` has at least `stride * (i + 1)` elements allocated.
        unsafe { std::slice::from_raw_parts(self.values.add(self.stride * i), self.stride) }
    }

    /// The i-th element as a mutable slice of length `stride`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `i < self.len()`.
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut [T] {
        // SAFETY: `values` has at least `stride * (i + 1)` elements allocated, and we have `&mut`.
        unsafe { std::slice::from_raw_parts_mut(self.values.add(self.stride * i), self.stride) }
    }

    /// Array view of the i-th element with the element shape.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `i < self.len()`.
    #[inline(always)]
    pub unsafe fn view_unchecked(&self, i: usize) -> ArrayViewD<'_, T> {
        // SAFETY: the shape is valid for the buffer which has length `stride`.
        unsafe { ArrayViewD::from_shape_ptr(IxDyn(self.shape()), self.get_unchecked(i).as_ptr()) }
    }

    /// Mutable array view of the i-th element with the element shape.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `i < self.len()`.
    #[inline(always)]
    pub unsafe fn view_unchecked_mut(&mut self, i: usize) -> ArrayViewMutD<'_, T> {
        // SAFETY: the shape is valid for the buffer which has length `stride`, and we have `&mut`.
        unsafe {
            ArrayViewMutD::from_shape_ptr(
                IxDyn(self.shape()),
                self.get_unchecked_mut(i).as_mut_ptr(),
            )
        }
    }

    /// The current value as a slice of length `stride`.
    #[inline(always)]
    pub fn current(&self) -> &[T] {
        // SAFETY: `len` is at least 1 due to the initial value, so `len - 1` is a valid index.
        unsafe { self.get_unchecked(self.len().unchecked_sub(1)) }
    }

    /// The current value as a mutable slice of length `stride`.
    #[inline(always)]
    pub fn current_mut(&mut self) -> &mut [T] {
        // SAFETY: `len` is at least 1 due to the initial value, so `len - 1` is a valid index.
        unsafe { self.get_unchecked_mut(self.len().unchecked_sub(1)) }
    }

    /// Array view of the current value with the element shape.
    #[inline(always)]
    pub fn current_view(&self) -> ArrayViewD<'_, T> {
        // SAFETY: `len` is at least 1 due to the initial value, so `len - 1` is a valid index.
        unsafe { self.view_unchecked(self.len().unchecked_sub(1)) }
    }

    /// Mutable array view of the current value with the element shape.
    #[inline(always)]
    pub fn current_view_mut(&mut self) -> ArrayViewMutD<'_, T> {
        // SAFETY: `len` is at least 1 due to the initial value, so `len - 1` is a valid index.
        unsafe { self.view_unchecked_mut(self.len().unchecked_sub(1)) }
    }
}

impl<T: Copy> std::ops::Index<usize> for Series<T> {
    type Output = [T];

    /// The i-th element as a slice of length `stride`.
    #[inline(always)]
    fn index(&self, i: usize) -> &[T] {
        assert!(i < self.len(), "index out of bounds");
        // SAFETY: bounds check ensures `i` is valid.
        unsafe { self.get_unchecked(i) }
    }
}

impl<T: Copy> std::ops::IndexMut<usize> for Series<T> {
    /// The i-th element as a mutable slice of length `stride`.
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut [T] {
        assert!(i < self.len(), "index out of bounds");
        // SAFETY: bounds check ensures `i` is valid.
        unsafe { self.get_unchecked_mut(i) }
    }
}

impl<T: Copy> Series<T> {
    /// Array view of the i-th element with the element shape.
    #[inline(always)]
    pub fn view(&self, i: usize) -> ArrayViewD<'_, T> {
        assert!(i < self.len(), "index out of bounds");
        // SAFETY: bounds check ensures `i` is valid.
        unsafe { self.view_unchecked(i) }
    }

    /// Mutable array view of the i-th element with the element shape.
    #[inline(always)]
    pub fn view_mut(&mut self, i: usize) -> ArrayViewMutD<'_, T> {
        assert!(i < self.len(), "index out of bounds");
        // SAFETY: bounds check ensures `i` is valid.
        unsafe { self.view_unchecked_mut(i) }
    }
}

impl<T: Copy> Series<T> {
    /// Return the index of the last element whose timestamp is `<= t`.
    #[inline]
    pub fn as_of(&self, t: i64) -> usize {
        // The first element should always satisfy the predicate.
        self.index().partition_point(|&ts| ts <= t) - 1
    }

    /// Append a `(timestamp, value)` pair.
    ///
    /// `value.len()` must equal the stride (product of shape dimensions).
    #[inline(always)]
    pub fn push(&mut self, timestamp: i64, value: &[T]) {
        assert!(value.len() == self.stride());
        assert!(timestamp >= self.index()[self.len() - 1]);
        // Ensure that we have room for one more element.
        if self.len_shape[0] == self.cap {
            self.grow();
        }
        // SAFETY: `len < cap` after a possible grow; both buffers are large enough.
        unsafe {
            let tp = self.index.add(self.len_shape[0]);
            let vp = self.values.add(self.stride * self.len_shape[0]);
            tp.write(timestamp);
            std::ptr::copy_nonoverlapping(value.as_ptr(), vp, self.stride);
        }
        self.len_shape[0] += 1;
    }

    /// Double the capacity.
    fn grow(&mut self) {
        let new_cap = self.cap * 2;
        self.index = realloc_buf(self.index, self.cap, new_cap);
        self.values = realloc_buf(self.values, self.stride * self.cap, self.stride * new_cap);
        self.cap = new_cap;
    }
}

impl<T: Copy> Drop for Series<T> {
    fn drop(&mut self) {
        dealloc_buf(self.index, self.cap);
        dealloc_buf(self.values, self.stride * self.cap);
    }
}

// SAFETY: Series owns its allocations exclusively; no interior sharing.
unsafe impl<T: Copy + Send> Send for Series<T> {}

// SAFETY: &Series only exposes shared reads of the owned buffers.
unsafe impl<T: Copy + Sync> Sync for Series<T> {}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

fn alloc_buf<T>(count: usize) -> *mut T {
    let layout = std::alloc::Layout::array::<T>(count).unwrap();
    // SAFETY: layout is non-zero (count >= 1 enforced by callers).
    let p = unsafe { std::alloc::alloc(layout) as *mut T };
    assert!(!p.is_null(), "allocation failed");
    p
}

fn realloc_buf<T>(ptr: *mut T, old_count: usize, new_count: usize) -> *mut T {
    let old_layout = std::alloc::Layout::array::<T>(old_count).unwrap();
    let new_layout = std::alloc::Layout::array::<T>(new_count).unwrap();
    // SAFETY: `ptr` was allocated with `old_layout`; `new_size >= old_size`.
    let p = unsafe { std::alloc::realloc(ptr as *mut u8, old_layout, new_layout.size()) as *mut T };
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
        let mut s = Series::new(&[], &[0.0]);
        // Initial value is element 0 (at timestamp i64::MIN).
        assert_eq!(s.len(), 1);
        assert_eq!(s.current(), &[0.0]);
        s.push(1, &[10.0]);
        s.push(2, &[20.0]);
        s.push(3, &[30.0]);
        assert_eq!(s.len(), 4);
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.index(), &[i64::MIN, 1, 2, 3]);
        assert_eq!(s.values(), &[0.0, 10.0, 20.0, 30.0]);
        assert_eq!(s.current_view().as_slice().unwrap(), &[30.0]);
    }

    #[test]
    fn strided_series() {
        let mut s = Series::new(&[3], &[0.0, 0.0, 0.0]);
        s.push(1, &[1.0, 2.0, 3.0]);
        s.push(2, &[4.0, 5.0, 6.0]);
        assert_eq!(s.len(), 3); // initial + 2 pushes
        assert_eq!(s.stride(), 3);
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.current_view().as_slice().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(s.values(), &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn grow_beyond_initial_capacity() {
        let mut s = Series::<i32>::new(&[], &[0]);
        for i in 0..100 {
            s.push(i as i64, &[i]);
        }
        assert_eq!(s.len(), 101); // initial + 100 pushes
        assert_eq!(s.current_view().as_slice().unwrap(), &[99]);
    }

    #[test]
    fn values_view_scalar() {
        let mut s = Series::new(&[], &[0.0]);
        s.push(1, &[10.0]);
        s.push(2, &[20.0]);
        s.push(3, &[30.0]);

        // Scalar → shape [len] (includes initial)
        let view = s.values_view();
        assert_eq!(view.shape(), &[4]);
        assert_eq!(view[[0]], 0.0); // initial
        assert_eq!(view[[1]], 10.0);
        assert_eq!(view[[3]], 30.0);
        // Zero-copy.
        assert_eq!(view.as_slice().unwrap().as_ptr(), s.values().as_ptr());
    }

    #[test]
    fn values_view_vector() {
        let mut s = Series::new(&[3], &[0.0, 0.0, 0.0]);
        s.push(1, &[1.0, 2.0, 3.0]);
        s.push(2, &[4.0, 5.0, 6.0]);

        // Vector → shape [len, 3] (includes initial row)
        let view = s.values_view();
        assert_eq!(view.shape(), &[3, 3]);
        assert_eq!(view[[1, 0]], 1.0);
        assert_eq!(view[[1, 2]], 3.0);
        assert_eq!(view[[2, 0]], 4.0);
        assert_eq!(view[[2, 2]], 6.0);
        assert_eq!(view.as_slice().unwrap().as_ptr(), s.values().as_ptr());
    }

    #[test]
    fn values_view_matrix() {
        let mut s = Series::new(&[2, 3], &[0.0; 6]);
        s.push(1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        s.push(2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        // Matrix → shape [len, 2, 3] (includes initial)
        let view = s.values_view();
        assert_eq!(view.shape(), &[3, 2, 3]);
        assert_eq!(view[[1, 0, 0]], 1.0);
        assert_eq!(view[[1, 1, 2]], 6.0);
        assert_eq!(view[[2, 0, 0]], 7.0);
        assert_eq!(view[[2, 1, 2]], 12.0);
        assert_eq!(view.as_slice().unwrap().as_ptr(), s.values().as_ptr());
    }

    #[test]
    fn timestamps_view() {
        let mut s = Series::new(&[2, 3], &[0.0; 6]);
        s.push(100, &[0.0; 6]);
        s.push(200, &[0.0; 6]);
        let ts = s.index_view();
        assert_eq!(ts.shape(), &[3]); // initial + 2 pushes
        assert_eq!(ts[0], i64::MIN); // initial timestamp
        assert_eq!(ts[1], 100);
        assert_eq!(ts[2], 200);
        assert_eq!(ts.as_slice().unwrap().as_ptr(), s.index().as_ptr());
    }

    #[test]
    fn get_element() {
        let mut s = Series::new(&[2], &[0.0, 0.0]);
        s.push(10, &[1.0, 2.0]);
        s.push(20, &[3.0, 4.0]);

        assert_eq!(s.view(0).as_slice().unwrap(), &[0.0, 0.0]);
        assert_eq!(s.view(1).as_slice().unwrap(), &[1.0, 2.0]);
        assert_eq!(s.view(2).as_slice().unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn as_of_exact_and_between() {
        let mut s = Series::new(&[], &[0.0]);
        s.push(10, &[1.0]);
        s.push(20, &[2.0]);
        s.push(30, &[3.0]);
        // Before any element (even before initial at i64::MIN is impossible,
        // but test the boundary just above initial).
        assert_eq!(s.as_of(i64::MIN), 0);
        // Exact matches.
        assert_eq!(s.as_of(10), 1);
        assert_eq!(s.as_of(20), 2);
        assert_eq!(s.as_of(30), 3);
        // Between timestamps — returns earlier.
        assert_eq!(s.as_of(15), 1);
        assert_eq!(s.as_of(25), 2);
        // Beyond last timestamp.
        assert_eq!(s.as_of(100), 3);
    }

    #[test]
    fn as_of_duplicate_timestamps() {
        let mut s = Series::new(&[], &[0.0]);
        s.push(10, &[1.0]);
        s.push(10, &[2.0]);
        s.push(10, &[3.0]);
        // Should return the last of the duplicates.
        assert_eq!(s.as_of(10), 3);
    }
}
