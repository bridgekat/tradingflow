//! Iterators over array scalars.

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
