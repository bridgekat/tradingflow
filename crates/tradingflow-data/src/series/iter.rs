//! Iterators over series elements.

use crate::{Array, ArrayView, Instant, Layout, Scalar, layout};

/// Owned iterator over series elements.
#[derive(Debug, Clone)]
pub struct SeriesIntoIter<T: Scalar, const N: usize> {
    pub(super) layout: layout::RowMajor<N>,
    pub(super) stride: usize, // Always equals `layout.len()`.
    pub(super) instants: std::vec::IntoIter<Instant>,
    pub(super) data: std::vec::IntoIter<T>,
}

impl<T: Scalar, const N: usize> Iterator for SeriesIntoIter<T, N> {
    type Item = (Instant, Array<T, N>);

    fn next(&mut self) -> Option<Self::Item> {
        let ins = self.instants.next()?;
        let data = self.data.by_ref().take(self.stride).collect();
        let value = unsafe { Array::from_parts_unchecked(self.layout.extents(), data) };
        Some((ins, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.instants.len();
        (n, Some(n))
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for SeriesIntoIter<T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for SeriesIntoIter<T, N> {}

/// Iterator over series view elements.
#[derive(Debug, Clone)]
pub struct SeriesIter<'a, T: Scalar, const N: usize> {
    pub(super) layout: layout::Strided<N>,
    pub(super) stride: usize,
    pub(super) instants: &'a [Instant],
    pub(super) data: &'a [T],
}

impl<'a, T: Scalar, const N: usize> Iterator for SeriesIter<'a, T, N> {
    type Item = (Instant, ArrayView<'a, T, N>);

    fn next(&mut self) -> Option<Self::Item> {
        let (&ins, ins_rest) = self.instants.split_first()?;
        self.instants = ins_rest;
        let (data, data_rest) = self.data.split_at(self.stride.min(self.data.len()));
        self.data = data_rest;
        // SAFETY: `data.len() >= self.layout.span()` by the view invariant.
        let view = unsafe { ArrayView::from_parts_unchecked(self.layout, data) };
        Some((ins, view))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.instants.len();
        (n, Some(n))
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for SeriesIter<'_, T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for SeriesIter<'_, T, N> {}
