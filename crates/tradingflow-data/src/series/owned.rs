//! Inherent impls for the owned, growable [`Series<T, N>`](Series): construction,
//! logical→physical index mapping, dimensions, length/retention state, bulk and
//! per-element access, temporal lookup, and mutation (append + front-trim).

use std::ops::Range;

use super::{Retention, Series, SeriesView};
use crate::{ArrayView, Instant, Scalar, Shape};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Create an empty series with the given element extents and retention
    /// bound.
    pub fn new(extents: [usize; N], retention: Retention) -> Self {
        Self {
            timestamps: Vec::new(),
            data: Vec::new(),
            shape: Shape::row_major(extents),
            base: 0,
            retention,
        }
    }

    /// Create an empty, unbounded series with the given element extents.
    pub fn new_unbounded(extents: [usize; N]) -> Self {
        Self::new(extents, Retention::UNBOUNDED)
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
