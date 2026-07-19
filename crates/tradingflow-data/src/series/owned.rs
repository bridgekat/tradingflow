use std::ops::Range;

use super::{Retention, SeriesIntoIter, SeriesIter};
use crate::{Array, ArrayView, Instant, Layout, Scalar, SeriesView, layout};

/// An owned time series, with row-major contiguous rank-`N` array elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Series<T: Scalar, const N: usize> {
    layout: layout::RowMajor<N>,
    stride: usize, // Always equals `layout.len()`.
    base: usize,
    timestamps: Vec<Instant>,
    data: Vec<T>,
    retention: Retention,
}

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Creates an empty series with the given element extents and retention
    /// bound.
    pub fn new(extents: [usize; N], retention: Retention) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        Self {
            layout,
            stride,
            base: 0,
            timestamps: Vec::new(),
            data: Vec::new(),
            retention,
        }
    }

    /// Creates an empty, unbounded series with the given element extents.
    pub fn new_unbounded(extents: [usize; N]) -> Self {
        Self::new(extents, Retention::unbounded())
    }

    /// Creates a series from element extents, timestamps, and a row-major
    /// contiguous buffer of packed elements.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != timestamps.len() * extents.iter().product()`.
    pub fn from_parts(
        extents: [usize; N],
        timestamps: Vec<Instant>,
        data: Vec<T>,
        retention: Retention,
    ) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        assert_eq!(
            data.len(),
            timestamps.len() * stride,
            "from_parts: {} elements of stride {} expect {} scalars, got {}",
            timestamps.len(),
            stride,
            timestamps.len() * stride,
            data.len(),
        );
        Self {
            layout,
            stride,
            base: 0,
            timestamps,
            data,
            retention,
        }
    }

    /// Creates a series from element extents, timestamps, and a row-major
    /// contiguous buffer of packed elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that
    /// `data.len() == timestamps.len() * extents.iter().product()`.
    pub unsafe fn from_parts_unchecked(
        extents: [usize; N],
        timestamps: Vec<Instant>,
        data: Vec<T>,
        retention: Retention,
    ) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        debug_assert_eq!(data.len(), timestamps.len() * stride);
        Self {
            layout,
            stride,
            base: 0,
            timestamps,
            data,
            retention,
        }
    }

    /// Physical slot of logical element `i`, or `None` if `i` is outside the
    /// retained range.
    fn slot(&self, i: usize) -> Option<usize> {
        let range = self.range();
        if i >= range.start && i < range.end {
            Some(i - range.start)
        } else {
            None
        }
    }

    pub fn layout(&self) -> layout::RowMajor<N> {
        self.layout
    }

    pub fn timestamps(&self) -> &[Instant] {
        &self.timestamps
    }

    pub fn timestamps_mut(&mut self) -> &mut [Instant] {
        &mut self.timestamps
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn retention(&self) -> Retention {
        self.retention
    }

    pub fn retention_mut(&mut self) -> &mut Retention {
        &mut self.retention
    }

    /// The retained element index range `[base, len)`.
    pub fn range(&self) -> Range<usize> {
        self.base..(self.base + self.timestamps.len())
    }

    /// The number of retained elements.
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    /// Whether the series contains no retained elements.
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// Borrows the whole series as a [`SeriesView`].
    pub fn view(&self) -> SeriesView<'_, T, N> {
        // SAFETY: `self.data.len() == self.timestamps.len() * self.stride`
        // and `self.stride == self.layout.len() >= self.layout.span()`.
        unsafe {
            SeriesView::from_parts_unchecked(
                self.layout.into(),
                self.stride,
                &self.timestamps,
                &self.data,
            )
        }
    }

    /// Returns the element at logical index `i`.
    pub fn at(&self, i: usize) -> Option<(Instant, ArrayView<'_, T, N>)> {
        let i = self.slot(i)?;
        let ts = self.timestamps[i];
        let data = &self.data[i * self.stride..(i + 1) * self.stride];
        // SAFETY: `data.len() == self.stride == self.layout.len() >= self.layout.span()`.
        let view = unsafe { ArrayView::from_parts_unchecked(self.layout.into(), data) };
        Some((ts, view))
    }

    /// Appends an element to the series, possibly trimming the series.
    ///
    /// # Panics
    ///
    /// Panics if `value.extents() != self.layout().extents()`.
    pub fn push(&mut self, timestamp: Instant, value: ArrayView<'_, T, N>) {
        assert_eq!(
            value.extents(),
            self.layout().extents(),
            "push: extents mismatch",
        );
        self.data.extend_from_slice(&value.to_contiguous());
        self.timestamps.push(timestamp);
        self.maybe_trim();
    }

    /// Drops the oldest elements from the front of the series.
    ///
    /// Amortized `O(1)`: a front drain is `O(retained_len)`, so we only compact
    /// when we can reclaim at least half the buffer — paying each drain down
    /// over the next ~`retained_len / 2` pushes. Physical storage therefore
    /// stays within `2x` the retained window.
    fn maybe_trim(&mut self) {
        if self.retention.is_unbounded() {
            return;
        }
        let range = self.range();
        let mut keep_from = range.end;
        if let Some(c) = self.retention.count {
            keep_from = keep_from.min(range.end.saturating_sub(c.max(1)));
        }
        if let Some(d) = self.retention.duration {
            let cutoff = self.timestamps[range.len() - 1] - d;
            let p = self.timestamps.partition_point(|&t| t < cutoff);
            keep_from = keep_from.min(self.base + p);
        }
        let droppable = keep_from.saturating_sub(self.base);
        if droppable > 0 && droppable * 2 >= range.len() {
            self.data.drain(0..droppable * self.stride);
            self.timestamps.drain(0..droppable);
            self.base += droppable;
        }
    }

    /// Returns a new series with the specified element extents, without
    /// reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different scalar count.
    pub fn reshape<const M: usize>(self, extents: [usize; M]) -> Series<T, M> {
        let layout = layout::RowMajor::new(extents);
        assert_eq!(
            self.layout().len(),
            layout.len(),
            "reshape: current element len {} != new extents {:?} ({} scalars)",
            self.layout().len(),
            extents,
            layout.len(),
        );
        Series {
            layout,
            stride: self.stride,
            base: self.base,
            timestamps: self.timestamps,
            data: self.data,
            retention: self.retention,
        }
    }

    /// The whole retained range as a rank-`M` [`Array`], with axis 0 being the
    /// time axis.
    ///
    /// # Panics
    ///
    /// Panics if `M != N + 1`.
    ///
    /// `M` needs to be spelled explicitly because stable Rust cannot form
    /// `N + 1` in a type, so the relation is asserted at runtime.
    pub fn to_array<const M: usize>(self) -> Array<T, M> {
        assert_eq!(M, N + 1, "to_array: M ({M}) must be N + 1 ({})", N + 1);
        let mut extents = [0; M];
        extents[0] = self.len();
        extents[1..].copy_from_slice(&self.layout.extents());
        let layout = layout::RowMajor::new(extents);
        // SAFETY: `self.data.len() == self.retained_len() * self.layout.len()`
        unsafe { Array::from_parts_unchecked(layout.extents(), self.data.into()) }
    }
}

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Iterates over the elements.
    pub fn iter(&self) -> SeriesIter<'_, T, N> {
        self.view().iter()
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for &'a Series<T, N> {
    type Item = (Instant, ArrayView<'a, T, N>);
    type IntoIter = SeriesIter<'a, T, N>;

    /// Iterates over the elements.
    fn into_iter(self) -> Self::IntoIter {
        self.view().iter()
    }
}

impl<T: Scalar, const N: usize> IntoIterator for Series<T, N> {
    type Item = (Instant, Array<T, N>);
    type IntoIter = SeriesIntoIter<T, N>;

    /// Iterates over the elements, consuming the series.
    fn into_iter(self) -> Self::IntoIter {
        SeriesIntoIter {
            layout: self.layout,
            stride: self.stride,
            timestamps: self.timestamps.into_iter(),
            data: self.data.into_iter(),
        }
    }
}

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
    /// Returns `range().end` if all retained timestamps are less than
    /// `query_ts`, and `range().start` if `query_ts` precedes the retained
    /// window.
    pub fn search(&self, query_ts: Instant) -> usize {
        self.base + self.view().search(query_ts)
    }
}
