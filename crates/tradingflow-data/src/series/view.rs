//! Inherent impls and conversions for the borrowed [`SeriesView<'a, T, N>`](SeriesView):
//! construction, dimensions, length, bulk and per-element access, temporal
//! lookup, and materialization into an [`Array`] / [`Series`].

use std::ops::Range;

use super::{Series, SeriesView};
use crate::{Array, ArrayView, Instant, Scalar, Shape};

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
