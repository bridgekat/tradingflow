use crate::{Array, ArrayView, Instant, Layout, Scalar, layout};

/// Owned iterator over series elements.
#[derive(Debug, Clone)]
pub struct SeriesIntoIter<T: Scalar, const N: usize> {
    pub(super) layout: layout::RowMajor<N>,
    pub(super) stride: usize, // Always equals `layout.len()`.
    pub(super) timestamps: std::vec::IntoIter<Instant>,
    pub(super) data: std::vec::IntoIter<T>,
}

impl<T: Scalar, const N: usize> Iterator for SeriesIntoIter<T, N> {
    type Item = (Instant, Array<T, N>);

    fn next(&mut self) -> Option<Self::Item> {
        let ts = self.timestamps.next()?;
        let data = self.data.by_ref().take(self.stride).collect();
        let value = unsafe { Array::from_parts_unchecked(self.layout.extents(), data) };
        Some((ts, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.timestamps.len();
        (n, Some(n))
    }
}

impl<T: Scalar, const N: usize> DoubleEndedIterator for SeriesIntoIter<T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ts = self.timestamps.next_back()?;
        // Pull the last block's scalars off the back one by one (they come out
        // reversed), then restore row-major order. NB: `.rev().take(n).rev()`
        // would yield the same block but skip-consume the *front* of the
        // buffer (`Take::next_back` advances past the untaken remainder),
        // destroying the elements a later call would need.
        let mut data: Box<[T]> = (0..self.stride)
            .map(|_| self.data.next_back().expect("packed series invariant"))
            .collect();
        data.reverse();
        let value = unsafe { Array::from_parts_unchecked(self.layout.extents(), data) };
        Some((ts, value))
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for SeriesIntoIter<T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for SeriesIntoIter<T, N> {}

/// Iterator over series view elements.
#[derive(Debug, Clone)]
pub struct SeriesIter<'a, T: Scalar, const N: usize> {
    pub(super) layout: layout::Strided<N>,
    pub(super) stride: usize,
    pub(super) timestamps: &'a [Instant],
    pub(super) data: &'a [T],
}

impl<'a, T: Scalar, const N: usize> Iterator for SeriesIter<'a, T, N> {
    type Item = (Instant, ArrayView<'a, T, N>);

    fn next(&mut self) -> Option<Self::Item> {
        let (&ts, ts_rest) = self.timestamps.split_first()?;
        self.timestamps = ts_rest;
        let (data, data_rest) = self.data.split_at(self.stride);
        self.data = data_rest;
        let view = unsafe { ArrayView::from_parts_unchecked(self.layout, data) };
        Some((ts, view))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.timestamps.len();
        (n, Some(n))
    }
}

impl<'a, T: Scalar, const N: usize> DoubleEndedIterator for SeriesIter<'a, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (&ts, ts_rest) = self.timestamps.split_last()?;
        self.timestamps = ts_rest;
        let (data_rest, data) = self.data.split_at(self.data.len() - self.stride);
        self.data = data_rest;
        let view = unsafe { ArrayView::from_parts_unchecked(self.layout, data) };
        Some((ts, view))
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for SeriesIter<'_, T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for SeriesIter<'_, T, N> {}
