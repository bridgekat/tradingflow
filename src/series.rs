//! Time series with append-only semantics and temporal lookups.

use ndarray::{ArrayViewD, IxDyn};

use crate::Scalar;

/// A time series of uniformly-shaped elements.
///
/// Backed by a flat `Vec<T>` with explicit capacity doubling.  The logical
/// length is tracked by the timestamps vector.
///
/// The series is append-only and unbounded.  Users manage compaction
/// externally if needed.
pub struct Series<T: Scalar> {
    /// Flat scalar buffer.  Allocated length = `capacity * stride`.
    /// Logical data occupies `0..len * stride`.
    data: Vec<T>,
    /// Non-decreasing timestamps; `timestamps.len()` is the logical length.
    timestamps: Vec<i64>,
    /// Element shape `[s0, s1, ...]` (without the time axis).
    shape: Box<[usize]>,
    /// Cached product of shape dimensions (>= 1).
    stride: usize,
    /// Number of allocated rows.
    capacity: usize,
}

// ===========================================================================
// Construction
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Create an empty series with the given element shape.
    pub fn new(shape: &[usize]) -> Self {
        let stride = shape.iter().product::<usize>();
        Self {
            data: Vec::new(),
            timestamps: Vec::new(),
            shape: shape.into(),
            stride,
            capacity: 0,
        }
    }

    /// Create an empty series with pre-allocated capacity.
    pub fn with_capacity(shape: &[usize], capacity: usize) -> Self {
        let stride = shape.iter().product::<usize>();
        Self {
            data: vec![T::default(); capacity * stride],
            timestamps: Vec::with_capacity(capacity),
            shape: shape.into(),
            stride,
            capacity,
        }
    }
}

// ===========================================================================
// Metadata
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Element shape (without the time axis).
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of scalars per element (product of element shape dimensions).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Number of timestamped elements (logical length).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the series is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Current allocated row capacity.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ===========================================================================
// Append
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Append an element with the given timestamp.
    ///
    /// # Panics
    ///
    /// Panics if `value.len() != self.stride()`.
    pub fn push(&mut self, timestamp: i64, value: &[T]) {
        let stride = self.stride;
        assert_eq!(
            value.len(),
            stride,
            "push: expected {} scalars, got {}",
            stride,
            value.len(),
        );
        let len = self.len();
        if len == self.capacity {
            self.grow();
        }
        let offset = len * stride;
        self.data[offset..offset + stride].clone_from_slice(value);
        self.timestamps.push(timestamp);
    }

    /// Double the capacity.
    fn grow(&mut self) {
        let new_cap = self.capacity * 2 + 1;
        self.data.resize(new_cap * self.stride, T::default());
        self.capacity = new_cap;
    }
}

// ===========================================================================
// Bulk access
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// All timestamps in the series.
    #[inline(always)]
    pub fn timestamps(&self) -> &[i64] {
        &self.timestamps
    }

    /// Logical values as a flat scalar slice (length = `len × stride`).
    #[inline(always)]
    pub fn values(&self) -> &[T] {
        &self.data[..self.len() * self.stride]
    }
}

// ===========================================================================
// Positional access
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Element at positional index `i` (0-based) as a flat slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    #[inline(always)]
    pub fn at(&self, i: usize) -> &[T] {
        assert!(
            i < self.len(),
            "index {i} out of bounds (len {})",
            self.len()
        );
        let s = self.stride;
        &self.data[i * s..(i + 1) * s]
    }

    /// The most recent element as a flat slice, or `None` if empty.
    #[inline(always)]
    pub fn last(&self) -> Option<&[T]> {
        if self.is_empty() {
            None
        } else {
            Some(self.at(self.len() - 1))
        }
    }

    /// Most recent timestamp, or `None` if empty.
    #[inline(always)]
    pub fn last_timestamp(&self) -> Option<i64> {
        self.timestamps.last().copied()
    }

    /// Values in `[start, end)` as a flat slice.
    ///
    /// # Panics
    ///
    /// Panics if `end > self.len()` or `start > end`.
    #[inline(always)]
    pub fn values_range(&self, start: usize, end: usize) -> &[T] {
        assert!(
            end <= self.len() && start <= end,
            "values_range: [{start}, {end}) out of bounds (len {})",
            self.len()
        );
        let s = self.stride;
        &self.data[start * s..end * s]
    }

    /// Last `n` elements as `(timestamps, flat_values)`.
    ///
    /// Returns `min(n, len)` elements.
    pub fn tail(&self, n: usize) -> (&[i64], &[T]) {
        let len = self.len();
        let n = n.min(len);
        let start = len - n;
        let s = self.stride;
        (&self.timestamps[start..], &self.data[start * s..len * s])
    }
}

// ===========================================================================
// Temporal access
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// As-of lookup: find the most recent element with `ts <= query_ts`.
    ///
    /// Returns `None` if no element satisfies the condition.
    pub fn asof(&self, query_ts: i64) -> Option<&[T]> {
        let idx = self.timestamps.partition_point(|&ts| ts <= query_ts);
        if idx == 0 {
            None
        } else {
            Some(self.at(idx - 1))
        }
    }

    /// Index of first timestamp `>= query_ts` (binary search).
    ///
    /// Returns `len` if all timestamps are less than `query_ts`.
    pub fn search(&self, query_ts: i64) -> usize {
        self.timestamps.partition_point(|&ts| ts < query_ts)
    }
}

// ===========================================================================
// Ndarray views
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Ndarray view of element at index `i`.
    pub fn row(&self, i: usize) -> ArrayViewD<'_, T> {
        ArrayViewD::from_shape(IxDyn(self.shape()), self.at(i)).unwrap()
    }

    /// Ndarray view of the logical series: shape `[len, s0, s1, ...]`.
    pub fn view(&self) -> ArrayViewD<'_, T> {
        let len = self.len();
        let mut full_shape = Vec::with_capacity(self.shape.len() + 1);
        full_shape.push(len);
        full_shape.extend_from_slice(&self.shape);
        ArrayViewD::from_shape(IxDyn(&full_shape), self.values()).unwrap()
    }
}

// ===========================================================================
// Iteration
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Iterate over `(timestamp, flat_element_slice)` pairs.
    pub fn iter(&self) -> SeriesIter<'_, T> {
        SeriesIter {
            timestamps: &self.timestamps,
            data: self.values(),
            stride: self.stride,
            pos: 0,
        }
    }
}

/// Iterator over `(i64, &[T])` pairs in a [`Series`].
pub struct SeriesIter<'a, T> {
    timestamps: &'a [i64],
    data: &'a [T],
    stride: usize,
    pos: usize,
}

impl<'a, T> Iterator for SeriesIter<'a, T> {
    type Item = (i64, &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.timestamps.len() {
            return None;
        }
        let i = self.pos;
        self.pos += 1;
        let s = self.stride;
        Some((self.timestamps[i], &self.data[i * s..(i + 1) * s]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.timestamps.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for SeriesIter<'a, T> {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_push_and_access() {
        let mut s = Series::<f64>::new(&[2]);
        assert!(s.is_empty());

        s.push(100, &[1.0, 2.0]);
        s.push(200, &[3.0, 4.0]);
        s.push(300, &[5.0, 6.0]);

        assert_eq!(s.len(), 3);
        assert_eq!(s.stride(), 2);
        assert_eq!(s.timestamps(), &[100, 200, 300]);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.last(), Some([5.0, 6.0].as_slice()));
        assert_eq!(s.at(0), &[1.0, 2.0]);
        assert_eq!(s.at(1), &[3.0, 4.0]);
    }

    #[test]
    fn series_scalar() {
        let mut s = Series::<f64>::new(&[]);
        assert_eq!(s.stride(), 1);

        s.push(1, &[10.0]);
        s.push(2, &[20.0]);

        assert_eq!(s.len(), 2);
        assert_eq!(s.at(0), &[10.0]);
        assert_eq!(s.last(), Some([20.0].as_slice()));
    }

    #[test]
    fn series_asof() {
        let mut s = Series::<f64>::new(&[]);
        s.push(100, &[1.0]);
        s.push(200, &[2.0]);
        s.push(300, &[3.0]);

        assert_eq!(s.asof(50), None);
        assert_eq!(s.asof(100), Some([1.0].as_slice()));
        assert_eq!(s.asof(150), Some([1.0].as_slice()));
        assert_eq!(s.asof(200), Some([2.0].as_slice()));
        assert_eq!(s.asof(250), Some([2.0].as_slice()));
        assert_eq!(s.asof(300), Some([3.0].as_slice()));
        assert_eq!(s.asof(999), Some([3.0].as_slice()));
    }

    #[test]
    #[should_panic(expected = "push: expected 2 scalars, got 3")]
    fn series_push_wrong_size() {
        let mut s = Series::<f64>::new(&[2]);
        s.push(1, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn series_ndarray_view() {
        let mut s = Series::<f64>::new(&[3]);
        s.push(1, &[1.0, 2.0, 3.0]);
        s.push(2, &[4.0, 5.0, 6.0]);

        let row0 = s.row(0);
        assert_eq!(row0.shape(), &[3]);
        assert_eq!(row0.as_slice().unwrap(), &[1.0, 2.0, 3.0]);

        let v = s.view();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn series_shape() {
        let s = Series::<f64>::new(&[3, 4]);
        assert_eq!(s.shape(), &[3, 4]);
        assert_eq!(s.stride(), 12);
    }

    // -- New method tests ----------------------------------------------------

    #[test]
    fn with_capacity() {
        let mut s = Series::<f64>::with_capacity(&[2], 10);
        assert_eq!(s.capacity(), 10);
        assert!(s.is_empty());

        // Push within capacity — no reallocation.
        let cap_before = s.capacity();
        for i in 0..10 {
            s.push(i, &[i as f64, (i * 10) as f64]);
        }
        assert_eq!(s.capacity(), cap_before);
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn last_timestamp() {
        let mut s = Series::<f64>::new(&[]);
        assert_eq!(s.last_timestamp(), None);

        s.push(100, &[1.0]);
        assert_eq!(s.last_timestamp(), Some(100));

        s.push(200, &[2.0]);
        assert_eq!(s.last_timestamp(), Some(200));
    }

    #[test]
    fn values_range() {
        let mut s = Series::<f64>::new(&[2]);
        s.push(100, &[1.0, 2.0]);
        s.push(200, &[3.0, 4.0]);
        s.push(300, &[5.0, 6.0]);

        assert_eq!(s.values_range(0, 2), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.values_range(1, 3), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.values_range(2, 3), &[5.0, 6.0]);
        assert_eq!(s.values_range(0, 0), &[] as &[f64]);
    }

    #[test]
    fn tail() {
        let mut s = Series::<f64>::new(&[]);
        s.push(100, &[1.0]);
        s.push(200, &[2.0]);
        s.push(300, &[3.0]);

        let (ts, vals) = s.tail(2);
        assert_eq!(ts, &[200, 300]);
        assert_eq!(vals, &[2.0, 3.0]);

        // n > len returns all
        let (ts, vals) = s.tail(100);
        assert_eq!(ts, &[100, 200, 300]);
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn search() {
        let mut s = Series::<f64>::new(&[]);
        s.push(100, &[1.0]);
        s.push(200, &[2.0]);
        s.push(300, &[3.0]);

        assert_eq!(s.search(50), 0);   // before all
        assert_eq!(s.search(100), 0);  // exact first
        assert_eq!(s.search(150), 1);  // between
        assert_eq!(s.search(200), 1);  // exact second
        assert_eq!(s.search(300), 2);  // exact last
        assert_eq!(s.search(999), 3);  // after all
    }

    #[test]
    fn iter() {
        let mut s = Series::<f64>::new(&[2]);
        s.push(100, &[1.0, 2.0]);
        s.push(200, &[3.0, 4.0]);

        let items: Vec<_> = s.iter().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], (100, [1.0, 2.0].as_slice()));
        assert_eq!(items[1], (200, [3.0, 4.0].as_slice()));
    }
}
