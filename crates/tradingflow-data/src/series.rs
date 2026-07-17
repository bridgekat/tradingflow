//! Time series with a **compile-time element rank** `N`, append-only
//! semantics, bounded retention, and temporal lookups — the [`Array`]-family
//! container for history.
//!
//! * [`Series<T, N>`] — owned, growable: a row-major contiguous value buffer
//!   plus one timestamp per element, growing along the (implicit) time axis.
//!   Lives in operator `State`, exactly like [`Array`].
//! * [`SeriesView<'a, T, N>`] — borrowed, `Copy`, self-contained window:
//!   `&[Instant]` + `&[T]` + the element [`Shape`]. The [`Series`] analogue of
//!   [`ArrayView`], convertible into a rank-`N + 1` [`ArrayView`] via
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
//! A series addresses its elements by **logical index** — the absolute
//! position since the series was created (`0` is the first element ever
//! pushed). [`len`](Series::len) is the logical length: the total number of
//! elements pushed, regardless of how many are still physically retained.
//!
//! A series may carry a [`Retention`] bound (a maximum element count and/or a
//! maximum time span). When set, [`push`](Series::push) drops the oldest
//! elements from the front so that physical storage stays bounded. A
//! [`base`](Series::base) offset records how many leading elements have been
//! dropped; positional accessors map a logical index `i` to the physical slot
//! `i - base`. The default retention is **unbounded** — nothing is ever
//! dropped, `base` stays `0`, and logical and physical indices coincide.
//!
//! Positional accessors ([`elem`](Series::elem),
//! [`timestamp`](Series::timestamp), [`window`](Series::window)) take logical
//! indices and panic if asked for an element that has been evicted
//! (`i < base`). The bulk accessors that return the whole retained window
//! ([`timestamps`](Series::timestamps), [`data`](Series::data),
//! [`view`](Series::view), [`tail`](Series::tail)) return only the window
//! `[base, len)`; for an unbounded series that is the full history. A
//! [`SeriesView`] is a plain window snapshot — **its indices are view-local**
//! (`0` is the view's first element), not logical.
//!
//! The view is also where reads are *defined*: every derived read on a
//! [`Series`] delegates to the corresponding [`SeriesView`] read on
//! [`view`](Series::view), after mapping logical indices to the retained
//! window. Within that window, a bounded series therefore reads identically
//! to an unbounded one by construction.
//!
//! # Views vs flat data
//!
//! Accessors hand out [`ArrayView`]/[`SeriesView`] borrows rather than flat
//! slices: [`elem`](Series::elem) and [`last`](Series::last) for one element,
//! [`view`](Series::view)/[`window`](Series::window)/[`tail`](Series::tail)
//! for a range, and [`push`](Series::push) to append one. The methods that
//! name `data` are the flat row-major escape hatches —
//! [`data`](Series::data)/[`data_mut`](Series::data_mut) for the whole
//! retained buffer, [`elem_data`](Series::elem_data) for one element, and
//! [`push_data`](Series::push_data) to append from a slice.

use std::ops::Range;

use super::array::{Array, ArrayView};
use super::time::{Duration, Instant};
use super::{Scalar, Shape};

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
    /// Physically retained timestamps (one per element).
    timestamps: Vec<Instant>,
    /// Physically retained elements (row-major, `shape.len()` scalars each).
    data: Vec<T>,
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

    /// Create an empty series with the given element extents and retention
    /// bound.
    pub fn with_retention(extents: [usize; N], retention: Retention) -> Self {
        Self {
            timestamps: Vec::new(),
            data: Vec::new(),
            shape: Shape::row_major(extents),
            base: 0,
            retention,
        }
    }

    /// Create an unbounded series from element extents, timestamps, and a flat
    /// row-major buffer of packed elements — the owning counterpart of
    /// [`SeriesView::from_parts`].
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != timestamps.len() * extents.iter().product()`.
    pub fn from_vec(extents: [usize; N], timestamps: Vec<Instant>, data: Vec<T>) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(
            data.len(),
            shape.blocks_len(timestamps.len()),
            "from_vec: {} elements of extents {:?} expect {} scalars, got {}",
            timestamps.len(),
            extents,
            shape.blocks_len(timestamps.len()),
            data.len(),
        );
        Self {
            timestamps,
            data,
            shape,
            base: 0,
            retention: Retention::UNBOUNDED,
        }
    }
}

// ---------------------------------------------------------------------------
// Logical -> physical mapping
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Physical slot of logical index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range `[base, len)` (either not yet
    /// pushed, or already dropped by the retention bound).
    fn slot(&self, i: usize) -> usize {
        let len = self.len();
        assert!(
            i >= self.base && i < len,
            "index {i} out of retained range [{}, {len})",
            self.base
        );
        i - self.base
    }

    /// Physical slot range of the logical range.
    ///
    /// # Panics
    ///
    /// Panics if `range.start < base`, `range.end > len`, or the range is
    /// inverted.
    fn slots(&self, range: Range<usize>) -> Range<usize> {
        let len = self.len();
        assert!(
            range.start >= self.base && range.end <= len && range.start <= range.end,
            "range [{}, {}) out of retained range [{}, {len})",
            range.start,
            range.end,
            self.base
        );
        range.start - self.base..range.end - self.base
    }
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// The shape shared by every element (per-axis extents and strides; always
    /// canonical row-major).
    pub fn elem_shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis extents of each element.
    pub fn elem_extents(&self) -> [usize; N] {
        self.shape.extents()
    }

    /// Number of scalars in each element (product of extents).
    pub fn elem_len(&self) -> usize {
        self.shape.len()
    }

    /// Whether there are no scalars in each element (some extent is zero).
    pub fn elem_is_empty(&self) -> bool {
        self.shape.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Length & retention state
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Logical length: the total number of elements ever pushed (including any
    /// since dropped by the retention bound).
    pub fn len(&self) -> usize {
        self.base + self.timestamps.len()
    }

    /// Whether the series has never had an element pushed.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Number of physically retained elements (`len - base`).
    pub fn retained_len(&self) -> usize {
        self.timestamps.len()
    }

    /// Logical index of the oldest retained element (count of dropped elements).
    pub fn base(&self) -> usize {
        self.base
    }

    /// The retention bound applied on [`push`](Self::push).
    pub fn retention(&self) -> Retention {
        self.retention
    }

    /// Set the retention bound. Does not retroactively trim already-stored
    /// history; the bound applies on the next [`push`](Self::push).
    pub fn set_retention(&mut self, retention: Retention) {
        self.retention = retention;
    }
}

// ---------------------------------------------------------------------------
// Bulk access (retained window [base, len))
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// The retained timestamps (logical `[base, len)`), one per element.
    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
    }

    /// Flat immutable slice of the retained elements, packed row-major
    /// (logical `[base, len)`). [`view`](Self::view) is the borrowing view.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Flat mutable slice of the retained elements, packed row-major
    /// (logical `[base, len)`).
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow the whole retained window as a [`SeriesView`].
    pub fn view(&self) -> SeriesView<'_, T, N> {
        SeriesView {
            timestamps: &self.timestamps,
            data: &self.data,
            shape: self.shape,
        }
    }

    /// Borrow the logical range as a [`SeriesView`].
    ///
    /// # Panics
    ///
    /// Panics if `range.start < base`, `range.end > len`, or the range is
    /// inverted.
    pub fn window(&self, range: Range<usize>) -> SeriesView<'_, T, N> {
        let slots = self.slots(range);
        self.view().window(slots)
    }

    /// Borrow the last `min(n, retained_len)` retained elements as a
    /// [`SeriesView`].
    pub fn tail(&self, n: usize) -> SeriesView<'_, T, N> {
        self.view().tail(n)
    }
}

// ---------------------------------------------------------------------------
// Element access (logical indices)
//
// Each read is the corresponding view-local [`SeriesView`] read after mapping
// logical indices through `slot` — the single definition of what a bounded
// series returns within its retained window.
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Element at logical index `i` as a rank-`N` [`ArrayView`] —
    /// [`elem_data`](Self::elem_data) is the flat counterpart.
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range `[base, len)` (either not yet
    /// pushed, or already dropped by the retention bound).
    pub fn elem(&self, i: usize) -> ArrayView<'_, T, N> {
        self.view().elem(self.slot(i))
    }

    /// Element at logical index `i` as a flat row-major slice.
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range `[base, len)`.
    pub fn elem_data(&self, i: usize) -> &[T] {
        self.view().elem_data(self.slot(i))
    }

    /// Timestamp of the element at logical index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range `[base, len)`.
    pub fn timestamp(&self, i: usize) -> Instant {
        self.view().timestamp(self.slot(i))
    }

    /// The most recent element as a rank-`N` [`ArrayView`], or `None` if the
    /// series is empty.
    pub fn last(&self) -> Option<ArrayView<'_, T, N>> {
        self.view().last()
    }

    /// The most recent timestamp, or `None` if the series is empty.
    pub fn last_timestamp(&self) -> Option<Instant> {
        self.view().last_timestamp()
    }
}

// ---------------------------------------------------------------------------
// Temporal lookup
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// As-of lookup over the retained window: the most recent element with
    /// `ts <= query_ts`.
    ///
    /// Returns `None` if no *retained* element satisfies the condition (an older
    /// qualifying element may have been dropped by the retention bound).
    pub fn asof(&self, query_ts: Instant) -> Option<ArrayView<'_, T, N>> {
        self.view().asof(query_ts)
    }

    /// Logical index of the first timestamp `>= query_ts` (binary search over
    /// the retained window).
    ///
    /// Returns `len` if all retained timestamps are less than `query_ts`, and
    /// `base` if `query_ts` precedes the retained window.
    pub fn search(&self, query_ts: Instant) -> usize {
        self.base + self.view().search(query_ts)
    }
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Append a rank-`N` [`ArrayView`] element (materialized row-major), then
    /// trim the front to honour the retention bound.
    ///
    /// # Panics
    ///
    /// Panics if `value.extents() != self.elem_extents()`.
    pub fn push(&mut self, timestamp: Instant, value: ArrayView<'_, T, N>) {
        assert_eq!(
            value.extents(),
            self.elem_extents(),
            "push: element extents mismatch",
        );
        self.data.extend_from_slice(&value.to_contiguous());
        self.timestamps.push(timestamp);
        self.maybe_trim();
    }

    /// Append an element from a flat row-major slice, then trim the front to
    /// honour the retention bound — the flat counterpart of [`push`](Self::push).
    ///
    /// # Panics
    ///
    /// Panics if `value.len() != self.elem_len()`.
    pub fn push_data(&mut self, timestamp: Instant, value: &[T]) {
        assert_eq!(
            value.len(),
            self.elem_len(),
            "push_data: expected {} scalars, got {}",
            self.elem_len(),
            value.len(),
        );
        self.data.extend_from_slice(value);
        self.timestamps.push(timestamp);
        self.maybe_trim();
    }

    /// Change the element extents in place (same rank), without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different scalar count.
    pub fn elem_reshape(&mut self, extents: [usize; N]) {
        let shape = Shape::row_major(extents);
        assert_eq!(
            self.elem_len(),
            shape.len(),
            "elem_reshape: current elem_len {} != new extents {:?} ({} scalars)",
            self.elem_len(),
            extents,
            shape.len(),
        );
        self.shape = shape;
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
            self.data.drain(self.shape.blocks_range(0..droppable));
            self.timestamps.drain(0..droppable);
            self.base += droppable;
        }
    }
}

// ===========================================================================
// SeriesView<'a, T, N> — borrowed, Copy, self-contained window
// ===========================================================================

/// A borrowed window of a [`Series`]: one timestamp slice, one packed
/// row-major value slice, and the element [`Shape`] — the [`ArrayView`]
/// analogue for history.
///
/// Indices are **view-local** (`0` is the view's first element); a view
/// carries no logical/retention bookkeeping. `Copy` (the payload is
/// references plus plain `usize`s) and fully lifetime-checked — a view cannot
/// outlive the series it borrows from. Convertible into a rank-`N + 1`
/// [`ArrayView`] via [`as_array_view`](Self::as_array_view) (elements are
/// packed, so the conversion is zero-copy).
#[derive(Debug)]
pub struct SeriesView<'a, T: Scalar, const N: usize> {
    /// One timestamp per element, non-decreasing.
    timestamps: &'a [Instant],
    /// Packed row-major elements: `timestamps.len() * shape.len()` scalars.
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

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// Build a window from the element [`Shape`] (canonical row-major), a
    /// timestamp slice, and a flat row-major buffer of packed elements — the
    /// borrowing counterpart of [`Series::from_vec`].
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != timestamps.len() * shape.len()`.
    pub fn from_parts(shape: Shape<N>, timestamps: &'a [Instant], data: &'a [T]) -> Self {
        assert_eq!(
            data.len(),
            shape.blocks_len(timestamps.len()),
            "from_parts: {} elements of {} scalars expect {} scalars, got {}",
            timestamps.len(),
            shape.len(),
            shape.blocks_len(timestamps.len()),
            data.len(),
        );
        Self {
            timestamps,
            data,
            shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> SeriesView<'_, T, N> {
    /// The shape shared by every element (per-axis extents and strides; always
    /// canonical row-major).
    pub fn elem_shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis extents of each element.
    pub fn elem_extents(&self) -> [usize; N] {
        self.shape.extents()
    }

    /// Number of scalars in each element (product of extents).
    pub fn elem_len(&self) -> usize {
        self.shape.len()
    }
}

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> SeriesView<'_, T, N> {
    /// Number of elements in the window.
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the window holds no elements.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Bulk access
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// The window's timestamps (one per element, non-decreasing).
    pub fn timestamps(&self) -> &'a [Instant] {
        self.timestamps
    }

    /// Flat slice of the window's elements, packed row-major
    /// (`len() * elem_len()` scalars).
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// Sub-window of the given element range.
    ///
    /// # Panics
    ///
    /// Panics if `range.end > len()` or the range is inverted.
    pub fn window(&self, range: Range<usize>) -> SeriesView<'a, T, N> {
        SeriesView {
            timestamps: &self.timestamps[range.clone()],
            data: &self.data[self.shape.blocks_range(range)],
            shape: self.shape,
        }
    }

    /// The last `min(n, len())` elements.
    pub fn tail(&self, n: usize) -> SeriesView<'a, T, N> {
        self.window(self.len() - n.min(self.len())..self.len())
    }
}

// ---------------------------------------------------------------------------
// Element access (view-local indices)
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// Element at view-local index `i` as a rank-`N` [`ArrayView`] —
    /// [`elem_data`](Self::elem_data) is the flat counterpart.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()`.
    pub fn elem(&self, i: usize) -> ArrayView<'a, T, N> {
        ArrayView::from_parts(self.shape, self.elem_data(i))
    }

    /// Element at view-local index `i` as a flat row-major slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()`.
    pub fn elem_data(&self, i: usize) -> &'a [T] {
        &self.data[self.shape.block_range(i)]
    }

    /// Timestamp of the element at view-local index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len()`.
    pub fn timestamp(&self, i: usize) -> Instant {
        self.timestamps[i]
    }

    /// The most recent element as a rank-`N` [`ArrayView`], or `None` if the
    /// window is empty.
    pub fn last(&self) -> Option<ArrayView<'a, T, N>> {
        (!self.is_empty()).then(|| self.elem(self.len() - 1))
    }

    /// The most recent timestamp, or `None` if the window is empty.
    pub fn last_timestamp(&self) -> Option<Instant> {
        self.timestamps.last().copied()
    }
}

// ---------------------------------------------------------------------------
// Temporal lookup
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// As-of lookup: the most recent element with `ts <= query_ts`, or `None`
    /// if the window starts after `query_ts`.
    pub fn asof(&self, query_ts: Instant) -> Option<ArrayView<'a, T, N>> {
        let p = self.timestamps.partition_point(|&ts| ts <= query_ts);
        if p == 0 { None } else { Some(self.elem(p - 1)) }
    }

    /// View-local index of the first timestamp `>= query_ts` (binary search);
    /// `len()` if all timestamps are earlier.
    pub fn search(&self, query_ts: Instant) -> usize {
        self.timestamps.partition_point(|&ts| ts < query_ts)
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// The whole window as a rank-`M = N + 1` [`ArrayView`] — the time axis
    /// becomes axis 0, extents `[len(), elem_extents()…]`. Zero-copy: elements
    /// are packed row-major, so the result is contiguous.
    ///
    /// (`M` is spelled explicitly because stable Rust cannot form `N + 1` in a
    /// type; the relation is asserted at runtime, like the rank-changing
    /// operators.)
    ///
    /// # Panics
    ///
    /// Panics if `M != N + 1`.
    pub fn as_array_view<const M: usize>(&self) -> ArrayView<'a, T, M> {
        ArrayView::from_parts(self.shape.stacked(self.len()), self.data)
    }

    /// Copy the window into an owned rank-`M = N + 1` [`Array`] (time axis 0).
    ///
    /// # Panics
    ///
    /// Panics if `M != N + 1`.
    pub fn to_array<const M: usize>(&self) -> Array<T, M> {
        self.as_array_view::<M>().to_array()
    }

    /// Copy the window into an owned [`Series`] of the same element rank `N`
    /// (unbounded retention) — the [`Series`] analogue of
    /// [`ArrayView::to_array`].
    pub fn to_series(&self) -> Series<T, N> {
        Series::from_vec(
            self.shape.extents(),
            self.timestamps.to_vec(),
            self.data.to_vec(),
        )
    }
}

impl<'a, T: Scalar, const N: usize> From<&'a Series<T, N>> for SeriesView<'a, T, N> {
    fn from(s: &'a Series<T, N>) -> Self {
        s.view()
    }
}

impl<T: Scalar, const N: usize> From<SeriesView<'_, T, N>> for Series<T, N> {
    fn from(v: SeriesView<'_, T, N>) -> Self {
        v.to_series()
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

        s.push_data(ts(100), &[1.0, 2.0]);
        s.push_data(ts(200), &[3.0, 4.0]);
        s.push_data(ts(300), &[5.0, 6.0]);

        assert_eq!(s.len(), 3);
        assert_eq!(s.elem_len(), 2);
        assert_eq!(s.timestamps(), &[ts(100), ts(200), ts(300)]);
        assert_eq!(s.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.last().unwrap().data(), &[5.0, 6.0]);
        assert_eq!(s.elem(0).data(), &[1.0, 2.0]);
        assert_eq!(s.elem(1).data(), &[3.0, 4.0]);
        assert_eq!(s.elem(1).to_vec(), vec![3.0, 4.0]);
        assert_eq!(s.elem_data(1), &[3.0, 4.0]);
    }

    #[test]
    fn series_scalar() {
        let mut s = Series::<f64, 0>::new([]);
        assert_eq!(s.elem_len(), 1);

        s.push_data(ts(1), &[10.0]);
        s.push_data(ts(2), &[20.0]);

        assert_eq!(s.len(), 2);
        assert_eq!(s.elem(0).data(), &[10.0]);
        assert_eq!(s.last().unwrap().data(), &[20.0]);
    }

    #[test]
    fn series_asof() {
        let mut s = Series::<f64, 0>::new([]);
        s.push_data(ts(100), &[1.0]);
        s.push_data(ts(200), &[2.0]);
        s.push_data(ts(300), &[3.0]);

        assert_eq!(s.asof(ts(50)).map(|v| v.data()), None);
        assert_eq!(s.asof(ts(100)).map(|v| v.data()), Some([1.0].as_slice()));
        assert_eq!(s.asof(ts(150)).map(|v| v.data()), Some([1.0].as_slice()));
        assert_eq!(s.asof(ts(200)).map(|v| v.data()), Some([2.0].as_slice()));
        assert_eq!(s.asof(ts(250)).map(|v| v.data()), Some([2.0].as_slice()));
        assert_eq!(s.asof(ts(300)).map(|v| v.data()), Some([3.0].as_slice()));
        assert_eq!(s.asof(ts(999)).map(|v| v.data()), Some([3.0].as_slice()));
    }

    #[test]
    #[should_panic(expected = "push_data: expected 2 scalars, got 3")]
    fn series_push_data_wrong_size() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push_data(ts(1), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn series_elem_shape() {
        let s = Series::<f64, 2>::new([3, 4]);
        assert_eq!(s.elem_extents(), [3, 4]);
        assert_eq!(s.elem_len(), 12);
        assert!(s.elem_shape().is_contiguous());
    }

    #[test]
    fn last_timestamp() {
        let mut s = Series::<f64, 0>::new([]);
        assert_eq!(s.last_timestamp(), None);

        s.push_data(ts(100), &[1.0]);
        assert_eq!(s.last_timestamp(), Some(ts(100)));

        s.push_data(ts(200), &[2.0]);
        assert_eq!(s.last_timestamp(), Some(ts(200)));
    }

    #[test]
    fn window_data_over_logical_range() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push_data(ts(100), &[1.0, 2.0]);
        s.push_data(ts(200), &[3.0, 4.0]);
        s.push_data(ts(300), &[5.0, 6.0]);

        assert_eq!(s.window(0..2).data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.window(1..3).data(), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.window(2..3).data(), &[5.0, 6.0]);
        assert_eq!(s.window(0..0).data(), &[] as &[f64]);
    }

    #[test]
    fn push_matches_push_data() {
        let mut a = Series::<f64, 1>::new([2]);
        let mut b = Series::<f64, 1>::new([2]);
        let row = Array::from_vec([2], vec![1.0, 2.0]);
        a.push_data(ts(100), row.data());
        b.push(ts(100), row.view());
        assert_eq!(a, b);
    }

    #[test]
    fn push_materializes_a_strided_view() {
        // A strided element must land packed row-major in the series.
        let panel = Array::from_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let col1 = ArrayView::from_parts(Shape::strided([2], [3]), &panel.data()[1..]);
        let mut s = Series::<f64, 1>::new([2]);
        s.push(ts(100), col1);
        assert_eq!(s.data(), &[2.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "push: element extents mismatch")]
    fn push_wrong_extents() {
        let mut s = Series::<f64, 1>::new([2]);
        let row = Array::from_vec([3], vec![1.0, 2.0, 3.0]);
        s.push(ts(1), row.view());
    }

    #[test]
    fn search() {
        let mut s = Series::<f64, 0>::new([]);
        s.push_data(ts(100), &[1.0]);
        s.push_data(ts(200), &[2.0]);
        s.push_data(ts(300), &[3.0]);

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
        s.push_data(ts(100), &[1.0, 2.0]);
        s.push_data(ts(200), &[3.0, 4.0]);
        s.push_data(ts(300), &[5.0, 6.0]);

        let v = s.view();
        assert_eq!(v.len(), 3);
        assert_eq!(v.elem_len(), 2);
        assert_eq!(v.timestamps(), &[ts(100), ts(200), ts(300)]);
        assert_eq!(v.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(v.elem(1).data(), &[3.0, 4.0]);
        assert_eq!(v.elem(2).to_vec(), vec![5.0, 6.0]);
        assert_eq!(v.elem_data(1), &[3.0, 4.0]);
        assert_eq!(v.timestamp(0), ts(100));
        assert_eq!(v.last().unwrap().data(), &[5.0, 6.0]);
        assert_eq!(v.last_timestamp(), Some(ts(300)));

        let w = v.window(1..3);
        assert_eq!(w.len(), 2);
        assert_eq!(w.timestamps(), &[ts(200), ts(300)]);
        assert_eq!(w.data(), &[3.0, 4.0, 5.0, 6.0]);

        let t = s.tail(2);
        assert_eq!(t.timestamps(), &[ts(200), ts(300)]);
        assert_eq!(t.data(), &[3.0, 4.0, 5.0, 6.0]);
        // n > len returns all
        assert_eq!(s.tail(100).len(), 3);

        // Series::window takes logical indices.
        let sw = s.window(1..2);
        assert_eq!(sw.timestamps(), &[ts(200)]);
        assert_eq!(sw.data(), &[3.0, 4.0]);
    }

    #[test]
    fn view_asof_and_search() {
        let mut s = Series::<f64, 0>::new([]);
        s.push_data(ts(100), &[1.0]);
        s.push_data(ts(200), &[2.0]);
        let v = s.view();
        assert_eq!(v.asof(ts(50)).map(|v| v.data()), None);
        assert_eq!(v.asof(ts(150)).map(|v| v.data()), Some([1.0].as_slice()));
        assert_eq!(v.search(ts(150)), 1);
        assert_eq!(v.search(ts(999)), 2);
    }

    #[test]
    fn view_as_array_view() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push_data(ts(100), &[1.0, 2.0]);
        s.push_data(ts(200), &[3.0, 4.0]);
        s.push_data(ts(300), &[5.0, 6.0]);

        // Whole window: [3, 2], contiguous.
        let av = s.view().as_array_view::<2>();
        assert_eq!(av.extents(), [3, 2]);
        assert!(av.as_slice().is_some());
        assert_eq!(av.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Sub-window converts too.
        let av = s.window(1..3).as_array_view::<2>();
        assert_eq!(av.extents(), [2, 2]);
        assert_eq!(av.to_vec(), vec![3.0, 4.0, 5.0, 6.0]);

        // Owned copy.
        let arr = s.tail(1).to_array::<2>();
        assert_eq!(arr.extents(), [1, 2]);
        assert_eq!(arr.data(), &[5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "M (3) must be N + 1 (2)")]
    fn view_as_array_view_wrong_rank() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push_data(ts(100), &[1.0, 2.0]);
        let _ = s.view().as_array_view::<3>();
    }

    #[test]
    fn view_from_parts_checks_len() {
        let tss = [ts(1), ts(2)];
        let vals = [1.0, 2.0, 3.0, 4.0];
        let v = SeriesView::<f64, 1>::from_parts(Shape::row_major([2]), &tss, &vals);
        assert_eq!(v.len(), 2);
        assert_eq!(v.elem(1).data(), &[3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "from_parts: 2 elements of 2 scalars expect 4 scalars, got 3")]
    fn view_from_parts_wrong_len() {
        let tss = [ts(1), ts(2)];
        let vals = [1.0, 2.0, 3.0];
        let _ = SeriesView::<f64, 1>::from_parts(Shape::row_major([2]), &tss, &vals);
    }

    #[test]
    fn view_to_series() {
        let mut s = Series::<f64, 1>::new([2]);
        s.push_data(ts(100), &[1.0, 2.0]);
        s.push_data(ts(200), &[3.0, 4.0]);
        s.push_data(ts(300), &[5.0, 6.0]);

        // Whole-window copy of an unbounded series equals the original.
        let owned = s.view().to_series();
        assert_eq!(owned, s);

        // A sub-window copies just its elements into a fresh, same-rank series.
        let sub = s.window(1..3).to_series();
        assert_eq!(sub.elem_extents(), [2]);
        assert_eq!(sub.timestamps(), &[ts(200), ts(300)]);
        assert_eq!(sub.data(), &[3.0, 4.0, 5.0, 6.0]);
    }

    // -- Retention -----------------------------------------------------------

    #[test]
    fn count_retention_bounds_storage_and_preserves_logical_reads() {
        // Keep the most recent 3 elements; push 10.
        let mut s = Series::<f64, 1>::with_retention([1], Retention::count(3));
        for i in 0..10 {
            s.push_data(ts((i + 1) * 100), &[i as f64]);
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
        assert_eq!(s.elem(7).data(), &[7.0]);
        assert_eq!(s.elem(8).data(), &[8.0]);
        assert_eq!(s.elem(9).data(), &[9.0]);
        assert_eq!(s.last().unwrap().data(), &[9.0]);
        assert_eq!(s.timestamp(9), ts(1000));
        assert_eq!(s.window(7..10).data(), &[7.0, 8.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "out of retained range")]
    fn count_retention_evicts_old_indices() {
        let mut s = Series::<f64, 1>::with_retention([1], Retention::count(3));
        for i in 0..10 {
            s.push_data(ts((i + 1) * 100), &[i as f64]);
        }
        // Index 0 was dropped long ago.
        let _ = s.elem(0).data();
    }

    #[test]
    fn duration_retention_keeps_time_window() {
        // Keep everything within 250ns of the latest; ticks are 100ns apart.
        let mut s =
            Series::<f64, 1>::with_retention([1], Retention::duration(Duration::from_nanos(250)));
        for i in 0..10 {
            s.push_data(ts((i + 1) * 100), &[i as f64]);
        }
        // Latest ts = 1000; cutoff = 750 → keep ts in {800, 900, 1000} = indices 7,8,9.
        assert_eq!(s.len(), 10);
        assert_eq!(s.elem(9).data(), &[9.0]);
        assert_eq!(s.elem(8).data(), &[8.0]);
        assert_eq!(s.elem(7).data(), &[7.0]);
        assert!(s.base() <= 7, "kept window too small: base {}", s.base());
    }

    #[test]
    fn asof_and_search_use_logical_indices_under_retention() {
        // Regression: `asof` once mixed a physical partition point into a
        // logical accessor, which broke as soon as retention evicted elements.
        let mut s = Series::<f64, 1>::with_retention([1], Retention::count(3));
        for i in 0..10 {
            s.push_data(ts((i + 1) * 100), &[i as f64]);
        }
        assert!(s.base() > 0, "retention must have evicted something");

        assert_eq!(s.asof(ts(1000)).unwrap().data(), &[9.0]);
        assert_eq!(s.asof(ts(850)).unwrap().data(), &[7.0]);
        // Before the retained window: None, though older elements once matched.
        assert_eq!(s.asof(ts(100)).map(|v| v.data()), None);

        // `search` returns logical indices: the first ts >= 850 is t900,
        // logically element 8.
        assert_eq!(s.search(ts(850)), 8);
        assert_eq!(s.search(ts(9999)), s.len());
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
            bounded.push_data(t, &row);
            unbounded.push_data(t, &row);
            assert_eq!(bounded.len(), unbounded.len());
            let lo = bounded.base();
            for j in lo..bounded.len() {
                assert_eq!(bounded.elem(j).data(), unbounded.elem(j).data());
                assert_eq!(bounded.timestamp(j), unbounded.timestamp(j));
            }
            assert_eq!(
                bounded.last().unwrap().data(),
                unbounded.last().unwrap().data()
            );
        }
    }
}
