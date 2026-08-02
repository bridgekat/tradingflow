use std::ops::Range;

use super::{SeriesIntoIter, SeriesIter};
use crate::{Array, ArrayView, Instant, Layout, Scalar, SeriesView, layout};

/// An owned time series, with row-major contiguous rank-`N` array elements.
#[derive(Debug, Clone)]
pub struct Series<T: Scalar, const N: usize> {
    layout: layout::RowMajor<N>,
    stride: usize, // Always equals `layout.len()`.
    instants: Vec<Instant>,
    data: Vec<T>,
    base: usize, // Logical index of the first retained element.
}

impl<T: Scalar, const N: usize> Series<T, N> {
    /// Creates an empty series with the given element extents.
    pub fn new(extents: [usize; N]) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        Self {
            layout,
            stride,
            instants: Vec::new(),
            data: Vec::new(),
            base: 0,
        }
    }

    /// Creates a series from element extents, instants, and a row-major
    /// contiguous buffer of packed elements.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != instants.len() * extents.iter().product()`.
    pub fn from_parts(
        extents: [usize; N],
        instants: Vec<Instant>,
        data: Vec<T>,
        base: usize,
    ) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        assert_eq!(
            data.len(),
            instants.len() * stride,
            "from_parts: {} elements of stride {} expect {} scalars, got {}",
            instants.len(),
            stride,
            instants.len() * stride,
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

    /// Creates a series from element extents, instants, and a row-major
    /// contiguous buffer of packed elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that
    /// `data.len() == instants.len() * extents.iter().product()`.
    pub unsafe fn from_parts_unchecked(
        extents: [usize; N],
        instants: Vec<Instant>,
        data: Vec<T>,
        base: usize,
    ) -> Self {
        let layout = layout::RowMajor::new(extents);
        let stride = layout.len();
        debug_assert_eq!(data.len(), instants.len() * stride);
        Self {
            layout,
            stride,
            instants,
            data,
            base,
        }
    }

    pub fn layout(&self) -> layout::RowMajor<N> {
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

    pub fn instants(&self) -> &[Instant] {
        &self.instants
    }

    pub fn instants_mut(&mut self) -> &mut [Instant] {
        &mut self.instants
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
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
    pub fn at(&self, i: usize) -> (Instant, ArrayView<'_, T, N>) {
        let range = self.range();
        assert!(
            i >= range.start && i < range.end,
            "at: index {} out of retained range {:?}",
            i,
            range
        );
        let i = i - range.start;
        let ts = self.instants[i];
        let data = &self.data[i * self.stride..];
        // SAFETY: `data.len() >= self.stride == self.layout.len() >= self.layout.span()`.
        let view = unsafe { ArrayView::from_parts_unchecked(self.layout.into(), data) };
        (ts, view)
    }

    /// Borrows the whole series as a [`SeriesView`].
    pub fn view(&self) -> SeriesView<'_, T, N> {
        // SAFETY: `self.data.len() == self.instants.len() * self.stride` and
        // `self.stride == self.layout.len()`.
        unsafe {
            SeriesView::from_slice_unchecked(
                self.layout.extents(),
                &self.instants,
                &self.data,
                self.base,
            )
        }
    }

    /// Borrows the whole series as a [`SeriesView`] with the specified extents.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different scalar count.
    pub fn view_reshape<const M: usize>(&self, extents: [usize; M]) -> SeriesView<'_, T, M> {
        let layout = layout::RowMajor::new(extents);
        assert_eq!(
            self.layout.len(),
            layout.len(),
            "view_reshape: current len {} != new extents {:?} ({} scalars)",
            self.layout.len(),
            extents,
            layout.len(),
        );
        // SAFETY: `self.data.len() == self.instants.len() * self.stride` and
        // `self.stride == self.layout.len() == layout.len()`.
        unsafe {
            SeriesView::from_slice_unchecked(
                layout.extents(),
                &self.instants,
                &self.data,
                self.base,
            )
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
        // SAFETY: `self.data.len() == self.instants.len() * self.stride` and
        // `self.stride == self.layout.len() == layout.len()`.
        unsafe { Series::from_parts_unchecked(extents, self.instants, self.data, self.base) }
    }

    /// Appends an element to the series.
    ///
    /// # Panics
    ///
    /// Panics if `value.extents() != self.extents()`.
    pub fn push(&mut self, instant: Instant, value: ArrayView<'_, T, N>) {
        assert_eq!(value.extents(), self.extents(), "push: extents mismatch",);
        self.instants.push(instant);
        self.data.extend_from_slice(&value.to_contiguous());
    }

    /// Drops the `count` oldest retained elements from the front of the
    /// series. Logical indices of the remaining elements remain unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `count > len()`.
    pub fn trim(&mut self, count: usize) {
        assert!(
            count <= self.len(),
            "trim: count {} > len {}",
            count,
            self.len()
        );
        if count > 0 {
            self.instants.drain(0..count);
            self.data.drain(0..count * self.stride);
            self.base += count;
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
        // SAFETY: `self.data.len() == self.len() * self.layout.len()`.
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
            instants: self.instants.into_iter(),
            data: self.data.into_iter(),
        }
    }
}

impl<T: Scalar + PartialEq, const N: usize> PartialEq for Series<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.instants == other.instants
            && self.extents() == other.extents()
            && self.data == other.data
    }
}

impl<T: Scalar + Eq, const N: usize> Eq for Series<T, N> {}

impl<T: Scalar, const N: usize> Series<T, N> {
    /// As-of lookup: the most recent element with `ts <= query_ts`, or `None`
    /// if the retained window starts after `query_ts`.
    pub fn asof(&self, query_ts: Instant) -> Option<ArrayView<'_, T, N>> {
        self.view().asof(query_ts)
    }
}

impl<T: Scalar, const N: usize> From<(Vec<Instant>, Vec<Array<T, N>>)> for Series<T, N> {
    fn from((instants, values): (Vec<Instant>, Vec<Array<T, N>>)) -> Self {
        let Some(first) = values.first() else {
            panic!("from: empty values vector");
        };
        let layout = first.layout();
        let stride = layout.len();
        let mut data = Vec::with_capacity(instants.len() * stride);
        for value in values {
            assert_eq!(
                value.layout(),
                layout,
                "from: value extents {:?} != first value extents {:?}",
                value.extents(),
                layout.extents()
            );
            data.extend_from_slice(value.data());
        }
        Self {
            layout,
            stride,
            instants,
            data,
            base: 0,
        }
    }
}
