//! Time series with a **compile-time element rank** `N`, append-only
//! semantics, bounded retention, and temporal lookups — the [`Array`]-family
//! container for history.
//!
//! * [`Series<T, N>`] — owned, growable: a row-major contiguous value buffer
//!   plus one timestamp per element, growing along the (implicit) time axis.
//!   Lives in operator `State`, exactly like [`Array`].
//! * [`SeriesView<'a, T, N>`] — borrowed, `Copy`, self-contained window:
//!   `&[Instant]` + `&[T]` + the element [`Shape`]. The [`Series`] analogue of
//!   [`ArrayView`], convertible into a rank-`N+1` [`ArrayView`] via
//!   [`as_array_view`](SeriesView::as_array_view) (the time axis becomes
//!   axis 0).
//!
//! The element rank `N` is static; the extents are dynamic. A series is
//! conceptually a `[time, extents…]` tensor whose axis 0 grows on
//! [`push`](Series::push) — but the storage stays `Vec`-backed (an [`Array`]
//! is a fixed-size snapshot; a series needs amortized append and front-trim).
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
//! Positional accessors ([`at`], [`values_range`], [`timestamp_at`], [`window`])
//! take logical indices and panic if asked for an element that has been evicted
//! (`i < base`). The bulk accessors that return the whole retained window
//! ([`values`], [`timestamps`], [`view`], [`tail`]) return only the window
//! `[base, len)`; for an unbounded series that is the full history. A
//! [`SeriesView`] is a plain window snapshot — **its indices are view-local**
//! (`0` is the view's first row), not logical.

use super::Scalar;
use super::array::{Array, ArrayView, Shape};
use super::time::{Duration, Instant};

// ===========================================================================
// Retention — how much history a series keeps
// ===========================================================================

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

// ===========================================================================
// Series<T, N> — owned, growable, row-major contiguous backing store
// ===========================================================================

/// A time series of rank-`N` array elements in row-major contiguous layout.
///
/// Timestamps are [`Instant`]s (SI nanoseconds since the TAI epoch) in
/// non-decreasing order. See the [module docs](self) for logical vs physical
/// indexing and bounded retention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Series<T: Scalar, const N: usize> {
    /// Physically retained elements (row-major, `stride` scalars each).
    data: Vec<T>,
    /// Physically retained timestamps (one per element).
    timestamps: Vec<Instant>,
    /// Element shape (canonical row-major; the time axis is implicit).
    shape: Shape<N>,
    /// Logical index of physical element `0` — the count of elements dropped
    /// off the front. `0` for an unbounded series.
    base: usize,
    /// Retention bound applied on [`push`](Self::push).
    retention: Retention,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Create an empty, unbounded series with the given element extents.
    pub fn new(extents: [usize; N]) -> Self {
        Self::with_retention(extents, Retention::UNBOUNDED)
    }

    /// Create an empty series with the given element extents and retention bound.
    pub fn with_retention(extents: [usize; N], retention: Retention) -> Self {
        Self {
            data: Vec::new(),
            timestamps: Vec::new(),
            shape: Shape::row_major(extents),
            base: 0,
            retention,
        }
    }

    /// Create an unbounded series from timestamp and flat row-major value
    /// vectors.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != timestamps.len() * extents.iter().product()`.
    pub fn from_vec(extents: [usize; N], timestamps: Vec<Instant>, values: Vec<T>) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(
            values.len(),
            timestamps.len() * shape.len(),
            "from_vec: expected values length {}, got {}",
            timestamps.len() * shape.len(),
            values.len()
        );
        Self {
            data: values,
            timestamps,
            shape,
            base: 0,
            retention: Retention::UNBOUNDED,
        }
    }
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// The element shape (extents + canonical strides; without the time axis).
    #[inline(always)]
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis element extents (without the time axis).
    #[inline(always)]
    pub fn extents(&self) -> [usize; N] {
        self.shape.extents()
    }

    /// Element rank (the compile-time rank `N`; without the time axis).
    #[inline(always)]
    pub const fn ndim(&self) -> usize {
        N
    }

    /// Number of scalars per element (product of element extents).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.shape.len()
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

// ---------------------------------------------------------------------------
// Bulk access (retained window [base, len))
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Flat immutable slice of the retained timestamps (logical `[base, len)`).
    #[inline(always)]
    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
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

    /// Borrow the whole retained window as a [`SeriesView`].
    #[inline(always)]
    pub fn view(&self) -> SeriesView<'_, T, N> {
        SeriesView {
            timestamps: &self.timestamps,
            data: &self.data,
            shape: self.shape,
        }
    }

    /// Borrow logical `[start, end)` as a [`SeriesView`].
    ///
    /// # Panics
    ///
    /// Panics if `start < base`, `end > len`, or `start > end`.
    pub fn window(&self, start: usize, end: usize) -> SeriesView<'_, T, N> {
        let len = self.len();
        assert!(
            start >= self.base && end <= len && start <= end,
            "window: [{start}, {end}) out of retained range [{}, {len})",
            self.base
        );
        let s = self.stride();
        let ps = start - self.base;
        let pe = end - self.base;
        SeriesView {
            timestamps: &self.timestamps[ps..pe],
            data: &self.data[ps * s..pe * s],
            shape: self.shape,
        }
    }

    /// Borrow the last `min(n, retained_len)` retained elements as a
    /// [`SeriesView`].
    pub fn tail(&self, n: usize) -> SeriesView<'_, T, N> {
        let plen = self.timestamps.len();
        let start = plen - n.min(plen);
        let s = self.stride();
        SeriesView {
            timestamps: &self.timestamps[start..],
            data: &self.data[start * s..],
            shape: self.shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Positional access (logical indices)
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
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
        let s = self.stride();
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
        let s = self.stride();
        &mut self.data[p * s..(p + 1) * s]
    }

    /// Element at logical index `i` as a rank-`N` [`ArrayView`].
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range `[base, len)`.
    #[inline(always)]
    pub fn element(&self, i: usize) -> ArrayView<'_, T, N> {
        ArrayView::from_parts(self.at(i), self.shape)
    }

    /// The most recent element as a flat slice, or `None` if empty.
    #[inline(always)]
    pub fn last(&self) -> Option<&[T]> {
        let plen = self.timestamps.len();
        if plen == 0 {
            None
        } else {
            let s = self.stride();
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
        let s = self.stride();
        let ps = start - self.base;
        let pe = end - self.base;
        &self.data[ps * s..pe * s]
    }
}

// ---------------------------------------------------------------------------
// Temporal access
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
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
            let s = self.stride();
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

// ---------------------------------------------------------------------------
// Append
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
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

    /// Append a rank-`N` [`ArrayView`] element (materialized row-major), then
    /// trim the front to honour the retention bound.
    ///
    /// # Panics
    ///
    /// Panics if `value.extents() != self.extents()`.
    #[inline]
    pub fn push_view(&mut self, timestamp: Instant, value: &ArrayView<'_, T, N>) {
        assert_eq!(
            value.extents(),
            self.extents(),
            "push_view: element extents mismatch",
        );
        self.data.extend_from_slice(&value.to_contiguous());
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
            let s = self.stride();
            self.data.drain(0..droppable * s);
            self.timestamps.drain(0..droppable);
            self.base += droppable;
        }
    }
}

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Change the element extents in place (same rank), without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different scalar count.
    pub fn reshape(&mut self, extents: [usize; N]) {
        let shape = Shape::row_major(extents);
        assert_eq!(
            self.stride(),
            shape.len(),
            "reshape: current stride {} != new extents {:?} ({} scalars)",
            self.stride(),
            extents,
            shape.len(),
        );
        self.shape = shape;
    }
}

// ===========================================================================
// SeriesView<'a, T, N> — borrowed, Copy, self-contained window
// ===========================================================================

/// A borrowed window of a [`Series`]: one timestamp slice, one packed
/// row-major value slice, and the element [`Shape`] — the [`ArrayView`]
/// analogue for history.
///
/// Indices are **view-local** (`0` is the view's first row); a view carries no
/// logical/retention bookkeeping. `Copy` (the payload is references + plain
/// `usize`s) and fully lifetime-checked — a view cannot outlive the series it
/// borrows from. Convertible into a rank-`N+1` [`ArrayView`] via
/// [`as_array_view`](Self::as_array_view) (rows are packed, so the conversion
/// is zero-copy).
#[derive(Debug)]
pub struct SeriesView<'a, T: Scalar, const N: usize> {
    /// One timestamp per row, non-decreasing.
    timestamps: &'a [Instant],
    /// Packed row-major values: `timestamps.len() * stride` scalars.
    data: &'a [T],
    /// Element shape (canonical row-major; the time axis is axis 0 of `data`).
    shape: Shape<N>,
}

// Manual (not derived) `Clone`/`Copy`: the view is references + plain `usize`s,
// so it is `Copy` regardless of whether `T` is. (Derived `Copy` would wrongly
// demand `T: Copy`, which e.g. `String` does not satisfy.)
impl<T: Scalar, const N: usize> Clone for SeriesView<'_, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Scalar, const N: usize> Copy for SeriesView<'_, T, N> {}

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// Build a view from parts: a timestamp slice, a packed row-major value
    /// slice, and the element [`Shape`] (canonical row-major).
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != timestamps.len() * shape.len()`.
    pub fn from_parts(timestamps: &'a [Instant], data: &'a [T], shape: Shape<N>) -> Self {
        assert_eq!(
            data.len(),
            timestamps.len() * shape.len(),
            "from_parts: expected values length {}, got {}",
            timestamps.len() * shape.len(),
            data.len()
        );
        Self {
            timestamps,
            data,
            shape,
        }
    }

    /// The element shape (extents + canonical strides; without the time axis).
    #[inline(always)]
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis element extents (without the time axis).
    #[inline(always)]
    pub fn extents(&self) -> [usize; N] {
        self.shape.extents()
    }

    /// Number of scalars per element (product of element extents).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.shape.len()
    }

    /// Number of rows in the window.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the window holds no rows.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// The window's timestamps (one per row, non-decreasing).
    #[inline(always)]
    pub fn timestamps(&self) -> &'a [Instant] {
        self.timestamps
    }

    /// The window's packed row-major values (`len() * stride()` scalars).
    #[inline(always)]
    pub fn values(&self) -> &'a [T] {
        self.data
    }

    /// Row `i` as a flat slice.
    #[inline(always)]
    pub fn at(&self, i: usize) -> &'a [T] {
        let s = self.shape.len();
        &self.data[i * s..(i + 1) * s]
    }

    /// Row `i` as a rank-`N` [`ArrayView`].
    #[inline(always)]
    pub fn element(&self, i: usize) -> ArrayView<'a, T, N> {
        ArrayView::from_parts(self.at(i), self.shape)
    }

    /// Timestamp of row `i`.
    #[inline(always)]
    pub fn timestamp_at(&self, i: usize) -> Instant {
        self.timestamps[i]
    }

    /// The last row as a flat slice, or `None` if the window is empty.
    #[inline(always)]
    pub fn last(&self) -> Option<&'a [T]> {
        if self.is_empty() {
            None
        } else {
            Some(self.at(self.len() - 1))
        }
    }

    /// The last row's timestamp, or `None` if the window is empty.
    #[inline(always)]
    pub fn last_timestamp(&self) -> Option<Instant> {
        self.timestamps.last().copied()
    }

    /// Sub-window of rows `[start, end)`.
    ///
    /// # Panics
    ///
    /// Panics if `start > end` or `end > len()`.
    pub fn window(&self, start: usize, end: usize) -> SeriesView<'a, T, N> {
        let s = self.shape.len();
        SeriesView {
            timestamps: &self.timestamps[start..end],
            data: &self.data[start * s..end * s],
            shape: self.shape,
        }
    }

    /// The last `min(n, len())` rows.
    pub fn tail(&self, n: usize) -> SeriesView<'a, T, N> {
        self.window(self.len() - n.min(self.len()), self.len())
    }

    /// As-of lookup: the most recent row with `ts <= query_ts`, or `None` if
    /// the window starts after `query_ts`.
    pub fn asof(&self, query_ts: Instant) -> Option<&'a [T]> {
        let p = self.timestamps.partition_point(|&ts| ts <= query_ts);
        if p == 0 { None } else { Some(self.at(p - 1)) }
    }

    /// Index of the first row with timestamp `>= query_ts` (binary search);
    /// `len()` if all rows are earlier.
    pub fn search(&self, query_ts: Instant) -> usize {
        self.timestamps.partition_point(|&ts| ts < query_ts)
    }

    /// The whole window as a rank-`M = N + 1` [`ArrayView`] — the time axis
    /// becomes axis 0, extents `[len(), extents()…]`. Zero-copy: rows are
    /// packed row-major, so the result is contiguous.
    ///
    /// (`M` is spelled explicitly because stable Rust cannot form `N + 1` in a
    /// type; the relation is asserted at runtime, like the rank-changing
    /// operators.)
    ///
    /// # Panics
    ///
    /// Panics if `M != N + 1`.
    pub fn as_array_view<const M: usize>(&self) -> ArrayView<'a, T, M> {
        assert_eq!(M, N + 1, "as_array_view: M ({M}) must be N + 1 ({})", N + 1);
        let mut extents = [0usize; M];
        extents[0] = self.timestamps.len();
        extents[1..].copy_from_slice(&self.shape.extents());
        ArrayView::from_parts(self.data, Shape::row_major(extents))
    }

    /// Copy the window into an owned rank-`M = N + 1` [`Array`] (time axis 0).
    ///
    /// # Panics
    ///
    /// Panics if `M != N + 1`.
    pub fn to_array<const M: usize>(&self) -> Array<T, M> {
        self.as_array_view::<M>().to_array()
    }
}

impl<'a, T: Scalar, const N: usize> From<&'a Series<T, N>> for SeriesView<'a, T, N> {
    #[inline(always)]
    fn from(s: &'a Series<T, N>) -> Self {
        s.view()
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
        let mut s = Series::<f64, 1>::new([2]);
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
        assert_eq!(s.element(1).to_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn series_scalar() {
        let mut s = Series::<f64, 0>::new([]);
        assert_eq!(s.stride(), 1);

        s.push(ts(1), &[10.0]);
        s.push(ts(2), &[20.0]);

        assert_eq!(s.len(), 2);
        assert_eq!(s.at(0), &[10.0]);
        assert_eq!(s.last(), Some([20.0].as_slice()));
    }

    #[test]
    fn series_asof() {
        let mut s = Series::<f64, 0>::new([]);
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
        let mut s = Series::<f64, 1>::new([2]);
        s.push(ts(1), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn series_shape() {
        let s = Series::<f64, 2>::new([3, 4]);
        assert_eq!(s.extents(), [3, 4]);
        assert_eq!(s.stride(), 12);
        assert_eq!(s.ndim(), 2);
    }

    #[test]
    fn last_timestamp() {
        let mut s = Series::<f64, 0>::new([]);
        assert_eq!(s.last_timestamp(), None);

        s.push(ts(100), &[1.0]);
        assert_eq!(s.last_timestamp(), Some(ts(100)));

        s.push(ts(200), &[2.0]);
        assert_eq!(s.last_timestamp(), Some(ts(200)));
    }

    #[test]
    fn values_range() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push(ts(100), &[1.0, 2.0]);
        s.push(ts(200), &[3.0, 4.0]);
        s.push(ts(300), &[5.0, 6.0]);

        assert_eq!(s.values_range(0, 2), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.values_range(1, 3), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.values_range(2, 3), &[5.0, 6.0]);
        assert_eq!(s.values_range(0, 0), &[] as &[f64]);
    }

    #[test]
    fn push_view_matches_push() {
        let mut a = Series::<f64, 1>::new([2]);
        let mut b = Series::<f64, 1>::new([2]);
        let row = Array::from_vec([2], vec![1.0, 2.0]);
        a.push(ts(100), row.as_slice());
        b.push_view(ts(100), &row.view());
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "push_view: element extents mismatch")]
    fn push_view_wrong_extents() {
        let mut s = Series::<f64, 1>::new([2]);
        let row = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
        s.push_view(ts(1), &row.view());
    }

    #[test]
    fn search() {
        let mut s = Series::<f64, 0>::new([]);
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

    // -- SeriesView ----------------------------------------------------------

    #[test]
    fn view_window_tail_and_elements() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push(ts(100), &[1.0, 2.0]);
        s.push(ts(200), &[3.0, 4.0]);
        s.push(ts(300), &[5.0, 6.0]);

        let v = s.view();
        assert_eq!(v.len(), 3);
        assert_eq!(v.stride(), 2);
        assert_eq!(v.timestamps(), &[ts(100), ts(200), ts(300)]);
        assert_eq!(v.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(v.at(1), &[3.0, 4.0]);
        assert_eq!(v.element(2).to_vec(), vec![5.0, 6.0]);
        assert_eq!(v.timestamp_at(0), ts(100));
        assert_eq!(v.last(), Some([5.0, 6.0].as_slice()));
        assert_eq!(v.last_timestamp(), Some(ts(300)));

        let w = v.window(1, 3);
        assert_eq!(w.len(), 2);
        assert_eq!(w.timestamps(), &[ts(200), ts(300)]);
        assert_eq!(w.values(), &[3.0, 4.0, 5.0, 6.0]);

        let t = s.tail(2);
        assert_eq!(t.timestamps(), &[ts(200), ts(300)]);
        assert_eq!(t.values(), &[3.0, 4.0, 5.0, 6.0]);
        // n > len returns all
        assert_eq!(s.tail(100).len(), 3);

        // Series::window takes logical indices.
        let sw = s.window(1, 2);
        assert_eq!(sw.timestamps(), &[ts(200)]);
        assert_eq!(sw.values(), &[3.0, 4.0]);
    }

    #[test]
    fn view_asof_and_search() {
        let mut s = Series::<f64, 0>::new([]);
        s.push(ts(100), &[1.0]);
        s.push(ts(200), &[2.0]);
        let v = s.view();
        assert_eq!(v.asof(ts(50)), None);
        assert_eq!(v.asof(ts(150)), Some([1.0].as_slice()));
        assert_eq!(v.search(ts(150)), 1);
        assert_eq!(v.search(ts(999)), 2);
    }

    #[test]
    fn view_as_array_view() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push(ts(100), &[1.0, 2.0]);
        s.push(ts(200), &[3.0, 4.0]);
        s.push(ts(300), &[5.0, 6.0]);

        // Whole window: [3, 2], contiguous.
        let av = s.view().as_array_view::<2>();
        assert_eq!(av.extents(), [3, 2]);
        assert!(av.contiguous_slice().is_some());
        assert_eq!(av.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Sub-window converts too.
        let av = s.window(1, 3).as_array_view::<2>();
        assert_eq!(av.extents(), [2, 2]);
        assert_eq!(av.to_vec(), vec![3.0, 4.0, 5.0, 6.0]);

        // Owned copy.
        let arr = s.tail(1).to_array::<2>();
        assert_eq!(arr.extents(), [1, 2]);
        assert_eq!(arr.as_slice(), &[5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "as_array_view: M (3) must be N + 1 (2)")]
    fn view_as_array_view_wrong_rank() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push(ts(100), &[1.0, 2.0]);
        let _ = s.view().as_array_view::<3>();
    }

    #[test]
    fn view_from_parts_checks_len() {
        let tss = [ts(1), ts(2)];
        let vals = [1.0, 2.0, 3.0, 4.0];
        let v = SeriesView::<f64, 1>::from_parts(&tss, &vals, Shape::row_major([2]));
        assert_eq!(v.len(), 2);
        assert_eq!(v.at(1), &[3.0, 4.0]);
    }

    // -- Retention -----------------------------------------------------------

    #[test]
    fn count_retention_bounds_storage_and_preserves_logical_reads() {
        // Keep the most recent 3 elements; push 10.
        let mut s = Series::<f64, 1>::with_retention([1], Retention::count(3));
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
        let mut s = Series::<f64, 1>::with_retention([1], Retention::count(3));
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
            Series::<f64, 1>::with_retention([1], Retention::duration(Duration::from_nanos(250)));
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
        let mut bounded = Series::<f64, 1>::with_retention([2], Retention::count(window));
        let mut unbounded = Series::<f64, 1>::new([2]);
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
