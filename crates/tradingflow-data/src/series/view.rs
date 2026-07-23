use std::borrow::Cow;
use std::ops::Range;

use super::{Series, SeriesIter};
use crate::{ArrayView, Instant, Layout, Scalar, layout};

/// A borrowed, windowed, strided view of a [`Series`].
#[derive(Debug)]
pub struct SeriesView<'a, T: Scalar, const N: usize> {
    layout: layout::Strided<N>,
    stride: usize,
    instants: &'a [Instant],
    data: &'a [T],
    base: usize, // Logical index of the first retained element.
}

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// Creates a series view from row-major contiguous slices.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != instants.len() * extents.iter().product()`.
    pub fn from_slice(
        extents: [usize; N],
        instants: &'a [Instant],
        data: &'a [T],
        base: usize,
    ) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        assert_eq!(
            data.len(),
            instants.len() * stride,
            "from_slice: {} elements of stride {} expect {} scalars, got {}",
            instants.len(),
            stride,
            instants.len() * stride,
            data.len(),
        );
        Self {
            layout: layout.into(),
            stride,
            instants,
            data,
            base,
        }
    }

    /// Creates a series view from row-major contiguous slices.
    ///
    /// # Safety
    ///
    /// The caller must ensure that
    /// `data.len() == instants.len() * extents.iter().product()`.
    pub unsafe fn from_slice_unchecked(
        extents: [usize; N],
        instants: &'a [Instant],
        data: &'a [T],
        base: usize,
    ) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        debug_assert_eq!(data.len(), instants.len() * stride);
        Self {
            layout: layout.into(),
            stride,
            instants,
            data,
            base,
        }
    }

    /// Creates a series view from strided slices.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < layout.span_ext(instants.len(), stride)`.
    pub fn from_parts(
        layout: layout::Strided<N>,
        stride: usize,
        instants: &'a [Instant],
        data: &'a [T],
        base: usize,
    ) -> Self {
        assert!(
            data.len() >= layout.span_ext(instants.len(), stride),
            "from_parts: {} elements of stride {} span {} scalars, got {}",
            instants.len(),
            stride,
            layout.span_ext(instants.len(), stride),
            data.len(),
        );
        Self {
            layout,
            stride,
            instants,
            data,
            base,
        }
    }

    /// Creates a series view from strided slices.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data.len() >= layout.span_ext(instants.len(), stride)`.
    pub unsafe fn from_parts_unchecked(
        layout: layout::Strided<N>,
        stride: usize,
        instants: &'a [Instant],
        data: &'a [T],
        base: usize,
    ) -> Self {
        debug_assert!(data.len() >= layout.span_ext(instants.len(), stride));
        Self {
            layout,
            stride,
            instants,
            data,
            base,
        }
    }

    pub fn layout(&self) -> layout::Strided<N> {
        self.layout
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    pub fn extents(&self) -> [usize; N] {
        self.layout.extents()
    }

    pub fn instants(&self) -> &'a [Instant] {
        self.instants
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// The retained element index range `[base, base + len)`.
    pub fn range(&self) -> Range<usize> {
        self.base..(self.base + self.instants.len())
    }

    /// The number of retained elements.
    pub fn len(&self) -> usize {
        self.instants.len()
    }

    /// Whether the series contains no retained elements.
    pub fn is_empty(&self) -> bool {
        self.instants.is_empty()
    }

    /// Returns the element at logical index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is outside the retained range.
    pub fn at(&self, i: usize) -> (Instant, ArrayView<'a, T, N>) {
        assert!(
            i >= self.range().start && i < self.range().end,
            "at: index {} out of retained range {:?}",
            i,
            self.range()
        );
        let i = i - self.range().start;
        let ts = self.instants[i];
        let data = &self.data[i * self.stride..];
        // SAFETY: `data.len() >= self.layout.span()` since
        // `self.data.len() >= self.layout.span_ext(self.instants.len(), self.stride)`.
        let view = unsafe { ArrayView::from_parts_unchecked(self.layout, data) };
        (ts, view)
    }

    /// A sub-window of the given element range. Preserves logical indices, so
    /// an element at logical index `i` of the window is the same element as
    /// the one at logical index `i` of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `range` is inverted or outside the retained range.
    pub fn window(&self, range: Range<usize>) -> SeriesView<'a, T, N> {
        assert!(
            range.start <= range.end
                && range.start >= self.range().start
                && range.end <= self.range().end,
            "window: window range {:?} out of bounds for self range {:?}",
            range,
            self.range()
        );
        let i = range.start - self.range().start;
        let j = range.end - self.range().start;
        // SAFETY: `j <= self.len()` implies that
        // `self.data.len() >= i * self.stride + self.layout.span_ext(j - i, self.stride)`.
        unsafe {
            SeriesView::from_parts_unchecked(
                self.layout,
                self.stride,
                &self.instants[i..j],
                &self.data[i * self.stride..],
                self.base + i,
            )
        }
    }

    /// A view whose elements are the sub-regions selected by `slices`.
    ///
    /// # Panics
    ///
    /// Panics if `slices` is out of bounds on any element axis.
    pub fn slice(&self, slices: impl layout::IntoSlices<N>) -> Self {
        let (offset, layout) = self.layout.slice(slices);
        // SAFETY: `self.data.len() >= self.layout.span_ext(...) >= offset + layout.span_ext(...)`.
        unsafe {
            SeriesView::from_parts_unchecked(
                layout,
                self.stride,
                self.instants,
                &self.data[offset..],
                self.base,
            )
        }
    }

    /// A view whose elements are the rank-`M` sub-regions selected by `slices`.
    ///
    /// # Panics
    ///
    /// Panics if `slices` does not consume exactly `N` axes or produce
    /// exactly `M` axes, or is out of bounds on any element axis.
    pub fn slice_reshape<const M: usize, const K: usize>(
        &self,
        slices: impl layout::IntoSliceReshapes<K>,
    ) -> SeriesView<'a, T, M> {
        let (offset, layout) = self.layout.slice_reshape(slices);
        // SAFETY: `self.data.len() >= self.layout.span_ext(...) >= offset + layout.span_ext(...)`.
        unsafe {
            SeriesView::from_parts_unchecked(
                layout,
                self.stride,
                self.instants,
                &self.data[offset..],
                self.base,
            )
        }
    }

    /// A zero-copy view with element axes permuted: axis `d` of the result is
    /// axis `perm[d]` of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `perm` is not a permutation of `0..N`.
    pub fn transpose(&self, perm: [usize; N]) -> Self {
        let layout = self.layout.transpose(perm);
        // SAFETY: `self.data.len() >= self.layout.span_ext(...) == layout.span_ext(...)`.
        unsafe {
            SeriesView::from_parts_unchecked(
                layout,
                self.stride,
                self.instants,
                self.data,
                self.base,
            )
        }
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
                for j in self.layout.iter() {
                    owned.push(self.data[i * self.stride + j].clone());
                }
            }
            Cow::Owned(owned)
        }
    }

    /// Copy the view into an owned, contiguous [`Series`].
    pub fn to_series(&self) -> Series<T, N> {
        // SAFETY: `to_contiguous()` returns a slice of length `self.len() * self.layout.len()`.
        unsafe {
            Series::from_parts_unchecked(
                self.layout.extents(),
                self.instants().into(),
                self.to_contiguous().into(),
                self.base,
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
        // SAFETY: `self.data.len() >= self.layout.span_ext(self.instants.len(), self.stride)`.
        unsafe { ArrayView::from_parts_unchecked(layout, self.data()) }
    }
}

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// Iterates over the elements.
    pub fn iter(&self) -> SeriesIter<'a, T, N> {
        SeriesIter {
            layout: self.layout,
            stride: self.stride,
            instants: self.instants,
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
        v.to_series()
    }
}

impl<T: Scalar, const N: usize> Clone for SeriesView<'_, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Scalar, const N: usize> Copy for SeriesView<'_, T, N> {}

impl<T: Scalar + PartialEq, const N: usize> PartialEq for SeriesView<'_, T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.instants == other.instants
            && self.extents() == other.extents()
            && self.iter().zip(other.iter()).all(|((_, a), (_, b))| a == b)
    }
}

impl<T: Scalar + Eq, const N: usize> Eq for SeriesView<'_, T, N> {}

impl<'a, T: Scalar, const N: usize> SeriesView<'a, T, N> {
    /// As-of lookup: the most recent element with `ts <= query_ts`, or `None`
    /// if the window starts after `query_ts`.
    pub fn asof(&self, query_ts: Instant) -> Option<ArrayView<'a, T, N>> {
        let p = self.instants.partition_point(|&ts| ts <= query_ts);
        if p == 0 {
            None
        } else {
            Some(self.at(self.base + p - 1).1)
        }
    }
}
