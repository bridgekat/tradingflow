use std::borrow::Cow;
use std::ops::Range;

use super::{Retention, Series, SeriesIter};
use crate::{ArrayView, Instant, Layout, Scalar, layout};

/// A borrowed, windowed, strided view of a [`Series`].
#[derive(Debug)]
pub struct SeriesView<'a, T: Scalar, const N: usize> {
    layout: layout::Strided<N>,
    stride: usize,
    timestamps: &'a [Instant],
    data: &'a [T],
}

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// Creates a series view from row-major contiguous slices.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != timestamps.len() * extents.iter().product()`.
    pub fn from_slice(extents: [usize; N], timestamps: &'a [Instant], data: &'a [T]) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        assert_eq!(
            data.len(),
            timestamps.len() * stride,
            "from_slice: {} elements of stride {} expect {} scalars, got {}",
            timestamps.len(),
            stride,
            timestamps.len() * stride,
            data.len(),
        );
        Self {
            layout: layout.into(),
            stride,
            timestamps,
            data,
        }
    }

    /// Creates a series view from row-major contiguous slices.
    ///
    /// # Safety
    ///
    /// The caller must ensure that
    /// `data.len() == timestamps.len() * extents.iter().product()`.
    pub unsafe fn from_slice_unchecked(
        extents: [usize; N],
        timestamps: &'a [Instant],
        data: &'a [T],
    ) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        debug_assert_eq!(data.len(), timestamps.len() * stride);
        Self {
            layout: layout.into(),
            stride,
            timestamps,
            data,
        }
    }

    /// Creates a series view from strided slices.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != timestamps.len() * stride` or
    /// `stride < layout.span()`.
    pub fn from_parts(
        layout: layout::Strided<N>,
        stride: usize,
        timestamps: &'a [Instant],
        data: &'a [T],
    ) -> Self {
        assert_eq!(
            data.len(),
            timestamps.len() * stride,
            "from_parts: {} elements of stride {} expect {} scalars, got {}",
            timestamps.len(),
            stride,
            timestamps.len() * stride,
            data.len(),
        );
        assert!(
            stride >= layout.span(),
            "from_parts: shape spans {} scalars, got {}",
            layout.span(),
            stride,
        );
        Self {
            layout,
            stride,
            timestamps,
            data,
        }
    }

    /// Creates a series view from strided slices.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data.len() == timestamps.len() * stride`
    /// and `stride >= layout.span()`.
    pub unsafe fn from_parts_unchecked(
        layout: layout::Strided<N>,
        stride: usize,
        timestamps: &'a [Instant],
        data: &'a [T],
    ) -> Self {
        debug_assert_eq!(data.len(), timestamps.len() * stride);
        debug_assert!(stride >= layout.span());
        Self {
            layout,
            stride,
            timestamps,
            data,
        }
    }

    pub fn layout(&self) -> layout::Strided<N> {
        self.layout
    }

    pub fn timestamps(&self) -> &'a [Instant] {
        self.timestamps
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    /// A sub-window of the given element range.
    ///
    /// # Panics
    ///
    /// Panics if `range.end > len()` or the range is inverted.
    pub fn window(&self, range: Range<usize>) -> SeriesView<'a, T, N> {
        SeriesView {
            layout: self.layout,
            stride: self.stride,
            timestamps: &self.timestamps[range.clone()],
            data: &self.data[range.start * self.stride..range.end * self.stride],
        }
    }

    /// Returns the element at index `i`.
    pub fn at(&self, i: usize) -> Option<(Instant, ArrayView<'a, T, N>)> {
        if i >= self.len() {
            return None;
        }
        let ts = self.timestamps[i];
        let data = &self.data[i * self.stride..(i + 1) * self.stride];
        // SAFETY: `data.len() == self.stride >= self.layout.span()`.
        let view = unsafe { ArrayView::from_parts_unchecked(self.layout, data) };
        Some((ts, view))
    }

    /// Returns `Some(data)` if the view has row-major contiguous packed layout,
    /// i.e. `layout.is_contiguous() && layout.len() == stride`.
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.layout.is_contiguous() && self.layout().len() == self.stride {
            Some(&self.data[..self.len() * self.layout.len()])
        } else {
            None
        }
    }

    /// Borrows the view's scalars as a row-major contiguous packed slice,
    /// materializing into an owned buffer if needed.
    pub fn to_contiguous(&self) -> Cow<'a, [T]> {
        if let Some(slice) = self.as_slice() {
            Cow::Borrowed(slice)
        } else {
            let mut owned = Vec::with_capacity(self.len() * self.layout.len());
            for i in 0..self.len() {
                for j in self.layout.offsets() {
                    owned.push(self.data[i * self.stride + j].clone());
                }
            }
            Cow::Owned(owned)
        }
    }

    /// Copy the view into an owned, contiguous [`Series`].
    pub fn to_series(&self, retention: Retention) -> Series<T, N> {
        // SAFETY: `to_contiguous()` returns a slice of length `self.len() * self.layout.len()`.
        unsafe {
            Series::from_parts_unchecked(
                self.layout.extents(),
                self.timestamps().into(),
                self.to_contiguous().into(),
                retention,
            )
        }
    }

    /// The whole window as a rank-`M` [`ArrayView`], with axis 0 being the
    /// time axis.
    ///
    /// # Panics
    ///
    /// Panics if `M != N + 1`.
    ///
    /// `M` needs to be spelled explicitly because stable Rust cannot form
    /// `N + 1` in a type, so the relation is asserted at runtime.
    pub fn to_array_view<const M: usize>(&self) -> ArrayView<'a, T, M> {
        assert_eq!(M, N + 1, "to_array_view: M ({M}) must be N + 1 ({})", N + 1);
        let mut extents = [0; M];
        let mut strides = [0; M];
        extents[0] = self.len();
        strides[0] = self.stride;
        extents[1..].copy_from_slice(&self.layout.extents());
        strides[1..].copy_from_slice(&self.layout.strides());
        let layout = layout::Strided::new(extents, strides);
        // SAFETY: `self.data.len() == self.len() * self.stride >= layout.span()`.
        unsafe { ArrayView::from_parts_unchecked(layout, self.data()) }
    }
}

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// Iterates over the elements.
    pub fn iter(&self) -> SeriesIter<'a, T, N> {
        SeriesIter {
            layout: self.layout,
            stride: self.stride,
            timestamps: self.timestamps,
            data: self.data,
        }
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for SeriesView<'a, T, N> {
    type Item = (Instant, ArrayView<'a, T, N>);
    type IntoIter = SeriesIter<'a, T, N>;

    /// Iterates over the elements.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Scalar, const N: usize> From<&'a Series<T, N>> for SeriesView<'a, T, N> {
    fn from(s: &'a Series<T, N>) -> Self {
        s.view()
    }
}

impl<T: Scalar, const N: usize> From<SeriesView<'_, T, N>> for Series<T, N> {
    fn from(v: SeriesView<'_, T, N>) -> Self {
        v.to_series(Retention::unbounded())
    }
}

impl<T: Scalar, const N: usize> Clone for SeriesView<'_, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Scalar, const N: usize> Copy for SeriesView<'_, T, N> {}

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// As-of lookup: the most recent element with `ts <= query_ts`, or `None`
    /// if the window starts after `query_ts`.
    pub fn asof(&self, query_ts: Instant) -> Option<ArrayView<'a, T, N>> {
        let p = self.timestamps.partition_point(|&ts| ts <= query_ts);
        if p == 0 {
            None
        } else {
            self.at(p - 1).map(|(_, v)| v)
        }
    }

    /// View-local index of the first timestamp `>= query_ts` (binary search);
    /// `len()` if all timestamps are earlier.
    pub fn search(&self, query_ts: Instant) -> usize {
        self.timestamps.partition_point(|&ts| ts < query_ts)
    }
}
