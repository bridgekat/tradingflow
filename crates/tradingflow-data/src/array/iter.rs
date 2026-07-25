//! Iterators over array scalars and sub-views.

use super::ArrayView;
use crate::layout::Strided;
use crate::{Offsets, Scalar};

/// Owned iterator over array scalars in row-major order.
#[derive(Debug, Clone)]
pub struct ArrayIntoIter<T: Scalar, const N: usize> {
    pub(super) inner: std::vec::IntoIter<T>,
}

impl<T: Scalar, const N: usize> Iterator for ArrayIntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for ArrayIntoIter<T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for ArrayIntoIter<T, N> {}

/// Iterator over array view scalars in row-major order.
#[derive(Debug, Clone)]
pub struct ArrayIter<'a, T: Scalar, const N: usize> {
    pub(super) offsets: Offsets<N>,
    pub(super) data: &'a [T],
}

impl<'a, T: Scalar, const N: usize> Iterator for ArrayIter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.offsets.next().map(|off| &self.data[off])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for ArrayIter<'_, T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for ArrayIter<'_, T, N> {}

/// Iterator over sub-views in row-major order.
#[derive(Debug, Clone)]
pub struct ArrayOuterIter<'a, T: Scalar, const K: usize, const M: usize> {
    pub(super) offsets: Offsets<K>,
    pub(super) layout: Strided<M>,
    pub(super) data: &'a [T],
}

impl<'a, T: Scalar, const K: usize, const M: usize> Iterator for ArrayOuterIter<'a, T, K, M> {
    type Item = ArrayView<'a, T, M>;

    fn next(&mut self) -> Option<ArrayView<'a, T, M>> {
        let offset = self.offsets.next()?;
        // SAFETY: `offset + layout.span() <= parent.span() <= data.len()`
        // for an iterator constructed from `parent.split_iter()`.
        Some(unsafe { ArrayView::from_parts_unchecked(self.layout, &self.data[offset..]) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl<T: Scalar, const K: usize, const M: usize> ExactSizeIterator for ArrayOuterIter<'_, T, K, M> {}
impl<T: Scalar, const K: usize, const M: usize> std::iter::FusedIterator
    for ArrayOuterIter<'_, T, K, M>
{
}
