//! Iteration over array scalars in row-major order.
//!
//! One borrowed traversal ([`ArrayIter`]: clone per scalar, strided via
//! [`Shape::offsets`](crate::Shape::offsets)) backs [`Array::iter`],
//! [`ArrayView::iter`], and the borrowing `IntoIterator`s. An owned [`Array`]
//! is always contiguous, so its by-value [`IntoIterator`] ([`ArrayIntoIter`])
//! just moves scalars out of the backing buffer. Every element is yielded by
//! value (`Item = T`).

use super::{Array, ArrayView};
use crate::{Offsets, Scalar};

/// Borrowed row-major scalar iterator over an [`ArrayView`] / [`Array`], cloning
/// each scalar; created by [`ArrayView::iter`] / [`Array::iter`] (or by
/// iterating a `&Array` or an `ArrayView`). Honours strides, so a strided view
/// (a column, a squeezed axis) still visits its scalars in logical row-major
/// order.
#[derive(Clone, Debug)]
pub struct ArrayIter<'a, T: Scalar, const N: usize> {
    /// Backing slice from the view's origin; `offsets` address into it.
    data: &'a [T],
    offsets: Offsets<N>,
}

impl<T: Scalar, const N: usize> Iterator for ArrayIter<'_, T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.offsets.next().map(|off| self.data[off].clone())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offsets.size_hint()
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for ArrayIter<'_, T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for ArrayIter<'_, T, N> {}

/// Owned row-major scalar iterator over an [`Array`], moving each scalar out;
/// created by `Array`'s [`IntoIterator`]. An owned array is contiguous, so this
/// is a thin wrapper over the backing buffer's iterator.
#[derive(Clone, Debug)]
pub struct ArrayIntoIter<T: Scalar, const N: usize> {
    inner: std::vec::IntoIter<T>,
}

impl<T: Scalar, const N: usize> Iterator for ArrayIntoIter<T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: Scalar, const N: usize> DoubleEndedIterator for ArrayIntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back()
    }
}

impl<T: Scalar, const N: usize> ExactSizeIterator for ArrayIntoIter<T, N> {}
impl<T: Scalar, const N: usize> std::iter::FusedIterator for ArrayIntoIter<T, N> {}

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// Iterate the view's scalars in row-major order, cloning each — strided
    /// views are walked through their strides. The borrowing counterpart to
    /// `Array`'s moving [`into_iter`](Array::into_iter); [`IntoIterator`] for
    /// the view does the same.
    pub fn iter(&self) -> ArrayIter<'a, T, N> {
        ArrayIter {
            data: self.data,
            offsets: self.shape.offsets(),
        }
    }
}

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Iterate the array's scalars in row-major order, cloning each — the
    /// borrowing counterpart of the moving [`into_iter`](Array::into_iter).
    pub fn iter(&self) -> ArrayIter<'_, T, N> {
        self.view().iter()
    }
}

impl<T: Scalar, const N: usize> IntoIterator for Array<T, N> {
    type Item = T;
    type IntoIter = ArrayIntoIter<T, N>;

    /// Move the array's scalars out, in row-major order.
    fn into_iter(self) -> Self::IntoIter {
        ArrayIntoIter {
            inner: Vec::from(self.data).into_iter(),
        }
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for &'a Array<T, N> {
    type Item = T;
    type IntoIter = ArrayIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for ArrayView<'a, T, N> {
    type Item = T;
    type IntoIter = ArrayIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
