//! Time series with append-only semantics, bounded retention, and temporal
//! lookups.
//!
//! # Logical vs physical indexing
//!
//! A series addresses its elements by **logical index** — the absolute position
//! since the series was created (`0` is the first element ever pushed). [`len`]
//! is the logical length: the total number of elements pushed, regardless of how
//! many are still physically retained.
//!
//! A series may carry a [`Retention`] bound (a maximum element count and/or a
//! maximum time span). When set, [`push`] drops the oldest elements from the
//! front so that physical storage stays bounded. A [`base`] offset records how
//! many leading elements have been dropped; positional accessors map a logical
//! index `i` to the physical slot `i - base`. The default retention is
//! **unbounded** — nothing is ever dropped, `base` stays `0`, and logical and
//! physical indices coincide (so all existing behaviour is unchanged).
//!
//! Positional accessors ([`at`], [`values_range`], [`timestamp_at`]) take logical
//! indices and panic if asked for an element that has been evicted (`i < base`).
//! The bulk accessors that return the whole backing buffer ([`values`],
//! [`timestamps`], [`view`], [`tail`]) return only the **retained window**
//! `[base, len)`; for an unbounded series that is the full history.
//!
//! [`len`]: Series::len
//! [`base`]: Series::base
//! [`push`]: Series::push
//! [`at`]: Series::at
//! [`values_range`]: Series::values_range
//! [`timestamp_at`]: Series::timestamp_at
//! [`values`]: Series::values
//! [`timestamps`]: Series::timestamps
//! [`view`]: Series::view
//! [`tail`]: Series::tail

use super::Scalar;
use super::time::{Duration, Instant};

/// A retention bound for a [`Series`]: how much history to keep.
///
/// A bound is the **union** of its active constraints — an element is retained
/// if *either* it is among the most-recent `count` elements *or* its timestamp
/// is within `duration` of the latest. This lets a single record feed both a
/// count-windowed consumer (e.g. `Lag(244)`, `RollingMean::count(252)`) and a
/// time-windowed one (e.g. `RollingMean::time_delta(365d)`): set both, and the
/// larger window wins. The default (both `None`) is unbounded — nothing is
/// dropped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Retention {
    /// Keep at least the most-recent `count` elements (`None` = no count bound).
    pub count: Option<usize>,
    /// Keep at least all elements within `duration` of the latest timestamp
    /// (`None` = no time bound).
    pub duration: Option<Duration>,
}

impl Retention {
    /// Unbounded retention: never drop anything (the default).
    pub const UNBOUNDED: Retention = Retention {
        count: None,
        duration: None,
    };

    /// Keep the most-recent `count` elements.
    pub fn count(count: usize) -> Self {
        Self {
            count: Some(count),
            duration: None,
        }
    }

    /// Keep all elements within `duration` of the latest timestamp.
    pub fn duration(duration: Duration) -> Self {
        Self {
            count: None,
            duration: Some(duration),
        }
    }

    /// Keep the union of a `count` window and a `duration` window.
    pub fn count_and_duration(count: usize, duration: Duration) -> Self {
        Self {
            count: Some(count),
            duration: Some(duration),
        }
    }

    /// Whether this bound retains everything (no trimming).
    #[inline(always)]
    pub fn is_unbounded(&self) -> bool {
        self.count.is_none() && self.duration.is_none()
    }
}

/// A time series of N-dimensional arrays in row-major contiguous layout.
///
/// Timestamps are [`Instant`]s (SI nanoseconds since the PTP epoch) in
/// non-decreasing order. See the [module docs](self) for logical vs physical
/// indexing and bounded retention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Series<T: Scalar> {
    /// Physically retained elements (row-major, `stride` scalars each).
    data: Vec<T>,
    /// Physically retained timestamps (one per element).
    timestamps: Vec<Instant>,
    shape: Box<[usize]>,
    stride: usize,
    /// Logical index of physical element `0` — the count of elements dropped
    /// off the front. `0` for an unbounded series.
    base: usize,
    /// Retention bound applied on [`push`](Self::push).
    retention: Retention,
}

// ===========================================================================
// Construction
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Create an empty, unbounded series with the given element shape.
    pub fn new(shape: &[usize]) -> Self {
        Self::with_retention(shape, Retention::UNBOUNDED)
    }

    /// Create an empty series with the given element shape and retention bound.
    pub fn with_retention(shape: &[usize], retention: Retention) -> Self {
        let stride = shape.iter().product::<usize>();
        Self {
            data: Vec::new(),
            timestamps: Vec::new(),
            shape: shape.into(),
            stride,
            base: 0,
            retention,
        }
    }

    /// Create an unbounded series from timestamp and flat value vectors.
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
            base: 0,
            retention: Retention::UNBOUNDED,
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

    /// Logical length: the total number of elements ever pushed (including any
    /// since dropped by the retention bound).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.base + self.timestamps.len()
    }

    /// Number of physically retained elements (`len - base`).
    #[inline(always)]
    pub fn retained_len(&self) -> usize {
        self.timestamps.len()
    }

    /// Logical index of the oldest retained element (count of dropped elements).
    #[inline(always)]
    pub fn base(&self) -> usize {
        self.base
    }

    /// The retention bound applied on [`push`](Self::push).
    #[inline(always)]
    pub fn retention(&self) -> Retention {
        self.retention
    }

    /// Set the retention bound. Does not retroactively trim already-stored
    /// history; the bound applies on the next [`push`](Self::push).
    #[inline(always)]
    pub fn set_retention(&mut self, retention: Retention) {
        self.retention = retention;
    }

    /// Whether the series has had any element pushed.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

// ===========================================================================
// Bulk access (retained window [base, len))
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Flat immutable slice of the retained timestamps (logical `[base, len)`).
    #[inline(always)]
    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
    }

    /// Flat mutable slice of the retained timestamps (logical `[base, len)`).
    #[inline(always)]
    pub fn timestamps_mut(&mut self) -> &mut [Instant] {
        &mut self.timestamps
    }

    /// Flat immutable slice of the retained elements (logical `[base, len)`).
    #[inline(always)]
    pub fn values(&self) -> &[T] {
        &self.data
    }

    /// Flat mutable slice of the retained elements (logical `[base, len)`).
    #[inline(always)]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// ===========================================================================
// Positional access (logical indices)
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Element at logical index `i` as a flat slice.
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range `[base, len)` (either not yet
    /// pushed, or already dropped by the retention bound).
    #[inline(always)]
    pub fn at(&self, i: usize) -> &[T] {
        let len = self.len();
        assert!(
            i >= self.base && i < len,
            "index {i} out of retained range [{}, {len})",
            self.base
        );
        let p = i - self.base;
        let s = self.stride;
        &self.data[p * s..(p + 1) * s]
    }

    /// Element at logical index `i` as a mutable flat slice.
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range `[base, len)`.
    #[inline(always)]
    pub fn at_mut(&mut self, i: usize) -> &mut [T] {
        let len = self.len();
        assert!(
            i >= self.base && i < len,
            "index {i} out of retained range [{}, {len})",
            self.base
        );
        let p = i - self.base;
        let s = self.stride;
        &mut self.data[p * s..(p + 1) * s]
    }

    /// The most recent element as a flat slice, or `None` if empty.
    #[inline(always)]
    pub fn last(&self) -> Option<&[T]> {
        let plen = self.timestamps.len();
        if plen == 0 {
            None
        } else {
            let s = self.stride;
            Some(&self.data[(plen - 1) * s..plen * s])
        }
    }

    /// Most recent timestamp, or `None` if empty.
    #[inline(always)]
    pub fn last_timestamp(&self) -> Option<Instant> {
        self.timestamps.last().copied()
    }

    /// Timestamp at logical index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range `[base, len)`.
    #[inline(always)]
    pub fn timestamp_at(&self, i: usize) -> Instant {
        let len = self.len();
        assert!(
            i >= self.base && i < len,
            "timestamp_at {i} out of retained range [{}, {len})",
            self.base
        );
        self.timestamps[i - self.base]
    }

    /// Values in logical `[start, end)` as a flat slice.
    ///
    /// # Panics
    ///
    /// Panics if `start < base`, `end > len`, or `start > end`.
    #[inline(always)]
    pub fn values_range(&self, start: usize, end: usize) -> &[T] {
        let len = self.len();
        assert!(
            start >= self.base && end <= len && start <= end,
            "values_range: [{start}, {end}) out of retained range [{}, {len})",
            self.base
        );
        let s = self.stride;
        let ps = start - self.base;
        let pe = end - self.base;
        &self.data[ps * s..pe * s]
    }

    /// Last `n` retained elements as `(timestamps, flat_values)`.
    ///
    /// Returns `min(n, retained_len)` elements.
    pub fn tail(&self, n: usize) -> (&[Instant], &[T]) {
        let plen = self.timestamps.len();
        let n = n.min(plen);
        let start = plen - n;
        let s = self.stride;
        (&self.timestamps[start..], &self.data[start * s..plen * s])
    }
}

// ===========================================================================
// Temporal access
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// As-of lookup over the retained window: the most recent element with
    /// `ts <= query_ts`.
    ///
    /// Returns `None` if no *retained* element satisfies the condition (an older
    /// qualifying element may have been dropped by the retention bound).
    pub fn asof(&self, query_ts: Instant) -> Option<&[T]> {
        let p = self.timestamps.partition_point(|&ts| ts <= query_ts);
        if p == 0 {
            None
        } else {
            let s = self.stride;
            Some(&self.data[(p - 1) * s..p * s])
        }
    }

    /// Logical index of the first timestamp `>= query_ts` (binary search over
    /// the retained window).
    ///
    /// Returns `len` if all retained timestamps are less than `query_ts`, and
    /// `base` if `query_ts` precedes the retained window.
    pub fn search(&self, query_ts: Instant) -> usize {
        self.base + self.timestamps.partition_point(|&ts| ts < query_ts)
    }
}

// ===========================================================================
// Append
// ===========================================================================

impl<T: Scalar> Series<T> {
    /// Append an element with the given timestamp, then trim the front to honour
    /// the retention bound.
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
        self.maybe_trim();
    }

    /// Drop the oldest elements from the front to honour [`retention`], in
    /// amortized `O(1)`: a front drain is `O(retained_len)`, so we only compact
    /// when we can reclaim at least half the buffer — paying each drain down
    /// over the next ~`retained_len / 2` pushes. Physical storage therefore
    /// stays within `2x` the retained window.
    ///
    /// [`retention`]: Self::retention
    #[inline]
    fn maybe_trim(&mut self) {
        if self.retention.is_unbounded() {
            return;
        }
        let plen = self.timestamps.len();
        if plen == 0 {
            return;
        }
        let len = self.base + plen; // logical length

        // Oldest logical index the (union of) active bounds requires keeping.
        let mut keep_from = len;
        if let Some(c) = self.retention.count {
            keep_from = keep_from.min(len.saturating_sub(c.max(1)));
        }
        if let Some(d) = self.retention.duration {
            let cutoff = self.timestamps[plen - 1] - d;
            let p = self.timestamps.partition_point(|&t| t < cutoff);
            keep_from = keep_from.min(self.base + p);
        }
        // Never drop the most recent element.
        keep_from = keep_from.min(len - 1);

        let droppable = keep_from.saturating_sub(self.base);
        if droppable > 0 && droppable * 2 >= plen {
            let s = self.stride;
            self.data.drain(0..droppable * s);
            self.timestamps.drain(0..droppable);
            self.base += droppable;
        }
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
    fn series_shape() {
        let s = Series::<f64>::new(&[3, 4]);
        assert_eq!(s.shape(), &[3, 4]);
        assert_eq!(s.stride(), 12);
    }

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

    // -- Retention -----------------------------------------------------------

    #[test]
    fn count_retention_bounds_storage_and_preserves_logical_reads() {
        // Keep the most recent 3 elements; push 10.
        let mut s = Series::<f64>::with_retention(&[1], Retention::count(3));
        for i in 0..10 {
            s.push(ts((i + 1) * 100), &[i as f64]);
        }
        // Logical length is the full count; physical storage is bounded (<= 2x
        // the window thanks to amortized compaction).
        assert_eq!(s.len(), 10);
        assert!(
            s.retained_len() <= 6,
            "retained {} > 2x window",
            s.retained_len()
        );
        assert!(s.retained_len() >= 3, "fewer than the window retained");
        assert_eq!(s.base(), s.len() - s.retained_len());

        // The required window [7, 10) reads identically to an unbounded series.
        assert_eq!(s.at(7), &[7.0]);
        assert_eq!(s.at(8), &[8.0]);
        assert_eq!(s.at(9), &[9.0]);
        assert_eq!(s.last(), Some([9.0].as_slice()));
        assert_eq!(s.timestamp_at(9), ts(1000));
        assert_eq!(s.values_range(7, 10), &[7.0, 8.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "out of retained range")]
    fn count_retention_evicts_old_indices() {
        let mut s = Series::<f64>::with_retention(&[1], Retention::count(3));
        for i in 0..10 {
            s.push(ts((i + 1) * 100), &[i as f64]);
        }
        // Index 0 was dropped long ago.
        let _ = s.at(0);
    }

    #[test]
    fn duration_retention_keeps_time_window() {
        // Keep everything within 250ns of the latest; ticks are 100ns apart.
        let mut s =
            Series::<f64>::with_retention(&[1], Retention::duration(Duration::from_nanos(250)));
        for i in 0..10 {
            s.push(ts((i + 1) * 100), &[i as f64]);
        }
        // Latest ts = 1000; cutoff = 750 → keep ts in {800, 900, 1000} = indices 7,8,9.
        assert_eq!(s.len(), 10);
        assert_eq!(s.at(9), &[9.0]);
        assert_eq!(s.at(8), &[8.0]);
        assert_eq!(s.at(7), &[7.0]);
        assert!(s.base() <= 7, "kept window too small: base {}", s.base());
    }

    #[test]
    fn bounded_matches_unbounded_within_window() {
        // A bounded series reads identically to an unbounded one for every index
        // that the bound retains — the equivalence the retention contract rests on.
        let window = 5usize;
        let mut bounded = Series::<f64>::with_retention(&[2], Retention::count(window));
        let mut unbounded = Series::<f64>::new(&[2]);
        for i in 0..40usize {
            let row = [i as f64, (i * 2) as f64];
            let t = ts((i as i64 + 1) * 10);
            bounded.push(t, &row);
            unbounded.push(t, &row);
            assert_eq!(bounded.len(), unbounded.len());
            let lo = bounded.base();
            for j in lo..bounded.len() {
                assert_eq!(bounded.at(j), unbounded.at(j), "value mismatch at {j}");
                assert_eq!(bounded.timestamp_at(j), unbounded.timestamp_at(j));
            }
            assert_eq!(bounded.last(), unbounded.last());
        }
    }
}
