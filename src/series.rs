//! Time series with append-only semantics and temporal lookups.

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

use crate::Scalar;
use crate::time::Instant;

/// A time series of N-dimensional arrays in row-major contiguous layout.
///
/// Timestamps are [`Instant`]s (SI nanoseconds since the PTP epoch) in
/// non-decreasing order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Series<T: Scalar> {
    data: Vec<T>,
    timestamps: Vec<Instant>,
    shape: Box<[usize]>,
    stride: usize,
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
        }
    }

    /// Create a series from timestamp and flat value vectors.
    pub fn from_vec(shape: &[usize], timestamps: Vec<Instant>, values: Vec<T>) -> Self {
        let stride = shape.iter().product::<usize>();
        assert_eq!(
            values.len(),
            timestamps.len() * stride,
            "from_vec: expected values length {}, got {}",
            timestamps.len() * stride,
            values.len()
        );
        Self {
            data: values,
            timestamps,
            shape: shape.into(),
            stride,
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

    /// Number of dimensions.
    #[inline(always)]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of scalars per element (product of element shape dimensions).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the series is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

// ===========================================================================
// Bulk access
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Flat immutable slice of all timestamps.
    #[inline(always)]
    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
    }

    /// Flat mutable slice of all timestamps.
    #[inline(always)]
    pub fn timestamps_mut(&mut self) -> &mut [Instant] {
        &mut self.timestamps
    }

    /// Flat immutable slice of all elements.
    #[inline(always)]
    pub fn values(&self) -> &[T] {
        &self.data
    }

    /// Flat mutable slice of all elements.
    #[inline(always)]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.data
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

    /// Element at positional index `i` (0-based) as a mutable flat slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    #[inline(always)]
    pub fn at_mut(&mut self, i: usize) -> &mut [T] {
        assert!(
            i < self.len(),
            "index {i} out of bounds (len {})",
            self.len()
        );
        let s = self.stride;
        &mut self.data[i * s..(i + 1) * s]
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
    pub fn last_timestamp(&self) -> Option<Instant> {
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
    pub fn tail(&self, n: usize) -> (&[Instant], &[T]) {
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
    pub fn asof(&self, query_ts: Instant) -> Option<&[T]> {
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
    pub fn search(&self, query_ts: Instant) -> usize {
        self.timestamps.partition_point(|&ts| ts < query_ts)
    }
}

// ===========================================================================
// Ndarray views
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Immutable ndarray view of element at index `i`.
    pub fn view_at(&self, i: usize) -> ArrayViewD<'_, T> {
        ArrayViewD::from_shape(IxDyn(&self.shape), self.at(i)).unwrap()
    }

    /// Mutable ndarray view of element at index `i`.
    pub fn view_at_mut(&mut self, i: usize) -> ArrayViewMutD<'_, T> {
        ArrayViewMutD::from_shape(IxDyn(&self.shape), self.at_mut(i)).unwrap()
    }

    /// Immutable ndarray view of the logical series: shape `[len, s0, s1, ...]`.
    pub fn view(&self) -> ArrayViewD<'_, T> {
        let len = self.len();
        let mut full_shape = Vec::with_capacity(self.shape.len() + 1);
        full_shape.push(len);
        full_shape.extend_from_slice(&self.shape);
        ArrayViewD::from_shape(IxDyn(&full_shape), self.values()).unwrap()
    }

    /// Mutable ndarray view of the logical series: shape `[len, s0, s1, ...]`.
    pub fn view_mut(&mut self) -> ArrayViewMutD<'_, T> {
        let len = self.len();
        let mut full_shape = Vec::with_capacity(self.shape.len() + 1);
        full_shape.push(len);
        full_shape.extend_from_slice(&self.shape);
        ArrayViewMutD::from_shape(IxDyn(&full_shape), self.values_mut()).unwrap()
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
    #[inline(always)]
    pub fn push(&mut self, timestamp: Instant, value: &[T]) {
        assert_eq!(
            value.len(),
            self.stride(),
            "push: expected {} scalars, got {}",
            self.stride(),
            value.len(),
        );
        self.data.extend_from_slice(value);
        self.timestamps.push(timestamp);
    }
}

// ===========================================================================
// Reshape
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Change the shape without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different stride.
    pub fn reshape(&mut self, new_shape: &[usize]) {
        let new_stride = new_shape.iter().product::<usize>();
        assert_eq!(
            self.stride(),
            new_stride,
            "reshape: current stride {} != new shape stride {}",
            self.stride(),
            new_stride,
        );
        self.shape = new_shape.into();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    #[test]
    fn series_push_and_access() {
        let mut s = Series::<f64>::new(&[2]);
        assert!(s.is_empty());

        s.push(ts(100), &[1.0, 2.0]);
        s.push(ts(200), &[3.0, 4.0]);
        s.push(ts(300), &[5.0, 6.0]);

        assert_eq!(s.len(), 3);
        assert_eq!(s.stride(), 2);
        assert_eq!(s.timestamps(), &[ts(100), ts(200), ts(300)]);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.last(), Some([5.0, 6.0].as_slice()));
        assert_eq!(s.at(0), &[1.0, 2.0]);
        assert_eq!(s.at(1), &[3.0, 4.0]);
    }

    #[test]
    fn series_scalar() {
        let mut s = Series::<f64>::new(&[]);
        assert_eq!(s.stride(), 1);

        s.push(ts(1), &[10.0]);
        s.push(ts(2), &[20.0]);

        assert_eq!(s.len(), 2);
        assert_eq!(s.at(0), &[10.0]);
        assert_eq!(s.last(), Some([20.0].as_slice()));
    }

    #[test]
    fn series_asof() {
        let mut s = Series::<f64>::new(&[]);
        s.push(ts(100), &[1.0]);
        s.push(ts(200), &[2.0]);
        s.push(ts(300), &[3.0]);

        assert_eq!(s.asof(ts(50)), None);
        assert_eq!(s.asof(ts(100)), Some([1.0].as_slice()));
        assert_eq!(s.asof(ts(150)), Some([1.0].as_slice()));
        assert_eq!(s.asof(ts(200)), Some([2.0].as_slice()));
        assert_eq!(s.asof(ts(250)), Some([2.0].as_slice()));
        assert_eq!(s.asof(ts(300)), Some([3.0].as_slice()));
        assert_eq!(s.asof(ts(999)), Some([3.0].as_slice()));
    }

    #[test]
    #[should_panic(expected = "push: expected 2 scalars, got 3")]
    fn series_push_wrong_size() {
        let mut s = Series::<f64>::new(&[2]);
        s.push(ts(1), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn series_ndarray_view() {
        let mut s = Series::<f64>::new(&[3]);
        s.push(ts(1), &[1.0, 2.0, 3.0]);
        s.push(ts(2), &[4.0, 5.0, 6.0]);

        let row0 = s.view_at(0);
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
    fn last_timestamp() {
        let mut s = Series::<f64>::new(&[]);
        assert_eq!(s.last_timestamp(), None);

        s.push(ts(100), &[1.0]);
        assert_eq!(s.last_timestamp(), Some(ts(100)));

        s.push(ts(200), &[2.0]);
        assert_eq!(s.last_timestamp(), Some(ts(200)));
    }

    #[test]
    fn values_range() {
        let mut s = Series::<f64>::new(&[2]);
        s.push(ts(100), &[1.0, 2.0]);
        s.push(ts(200), &[3.0, 4.0]);
        s.push(ts(300), &[5.0, 6.0]);

        assert_eq!(s.values_range(0, 2), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.values_range(1, 3), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.values_range(2, 3), &[5.0, 6.0]);
        assert_eq!(s.values_range(0, 0), &[] as &[f64]);
    }

    #[test]
    fn tail() {
        let mut s = Series::<f64>::new(&[]);
        s.push(ts(100), &[1.0]);
        s.push(ts(200), &[2.0]);
        s.push(ts(300), &[3.0]);

        let (tss, vals) = s.tail(2);
        assert_eq!(tss, &[ts(200), ts(300)]);
        assert_eq!(vals, &[2.0, 3.0]);

        // n > len returns all
        let (tss, vals) = s.tail(100);
        assert_eq!(tss, &[ts(100), ts(200), ts(300)]);
        assert_eq!(vals, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn search() {
        let mut s = Series::<f64>::new(&[]);
        s.push(ts(100), &[1.0]);
        s.push(ts(200), &[2.0]);
        s.push(ts(300), &[3.0]);

        assert_eq!(s.search(ts(50)), 0); // before all
        assert_eq!(s.search(ts(100)), 0); // exact first
        assert_eq!(s.search(ts(150)), 1); // between
        assert_eq!(s.search(ts(200)), 1); // exact second
        assert_eq!(s.search(ts(300)), 2); // exact last
        assert_eq!(s.search(ts(999)), 3); // after all
    }
}
