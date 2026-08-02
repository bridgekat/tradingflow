use std::ops::{Deref, DerefMut, Index, IndexMut};

use super::{ArrayIntoIter, ArrayIter, ArrayView};
use crate::{Layout, Scalar, layout};

/// An owned, row-major contiguous, rank-`N` array.
#[derive(Debug, Clone)]
pub struct Array<T: Scalar, const N: usize> {
    layout: layout::RowMajor<N>,
    data: Box<[T]>,
}

impl<T: Scalar> Array<T, 0> {
    /// Creates a rank-0 array holding one scalar.
    pub fn scalar(value: T) -> Self {
        Self {
            layout: layout::RowMajor::new([]),
            data: vec![value].into(),
        }
    }
}

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Creates an array filled with `value`.
    pub fn full(extents: [usize; N], value: T) -> Self {
        let layout = layout::RowMajor::new(extents);
        Self {
            layout,
            data: vec![value; layout.len()].into(),
        }
    }

    /// Creates an array filled with `T::default()` (0 for numeric types).
    pub fn zeros(extents: [usize; N]) -> Self {
        Self::full(extents, T::default())
    }

    /// Creates an array from a row-major contiguous buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != extents.iter().product()`.
    pub fn from_parts(extents: [usize; N], data: Box<[T]>) -> Self {
        let layout = layout::RowMajor::new(extents);
        assert_eq!(
            data.len(),
            layout.len(),
            "from_parts: extents {:?} expect {} scalars, got {}",
            extents,
            layout.len(),
            data.len(),
        );
        Self { layout, data }
    }

    /// Creates an array from a row-major contiguous buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data.len() == extents.iter().product()`.
    pub unsafe fn from_parts_unchecked(extents: [usize; N], data: Box<[T]>) -> Self {
        let layout = layout::RowMajor::new(extents);
        debug_assert_eq!(data.len(), layout.len());
        Self { layout, data }
    }

    pub fn layout(&self) -> layout::RowMajor<N> {
        self.layout
    }

    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    pub fn extents(&self) -> [usize; N] {
        self.layout.extents()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrows the whole array as an [`ArrayView`].
    pub fn view(&self) -> ArrayView<'_, T, N> {
        // SAFETY: `self.data.len() == self.layout.len()`.
        unsafe { ArrayView::from_slice_unchecked(self.layout.extents(), &self.data) }
    }

    /// Borrow the whole array as an [`ArrayView`] with the specified extents.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different scalar count.
    pub fn view_reshape<const M: usize>(&self, extents: [usize; M]) -> ArrayView<'_, T, M> {
        let layout = layout::RowMajor::new(extents);
        assert_eq!(
            self.layout.len(),
            layout.len(),
            "view_reshape: current len {} != new extents {:?} ({} scalars)",
            self.layout.len(),
            extents,
            layout.len(),
        );
        // SAFETY: `self.data.len() == self.layout.len() == layout.len()`.
        unsafe { ArrayView::from_slice_unchecked(layout.extents(), &self.data) }
    }

    /// Returns a new array with the specified extents, without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different scalar count.
    pub fn reshape<const M: usize>(self, extents: [usize; M]) -> Array<T, M> {
        let layout = layout::RowMajor::new(extents);
        assert_eq!(
            self.layout.len(),
            layout.len(),
            "reshape: current len {} != new extents {:?} ({} scalars)",
            self.layout.len(),
            extents,
            layout.len(),
        );
        // SAFETY: `self.data.len() == self.layout.len() == layout.len()`.
        unsafe { Array::from_parts_unchecked(extents, self.data) }
    }

    /// Writes data into the array.
    ///
    /// # Panics
    ///
    /// Panics if `value.extents() != self.extents()`.
    pub fn assign(&mut self, value: ArrayView<'_, T, N>) {
        assert_eq!(value.extents(), self.extents(), "assign: extents mismatch");
        if let Some(s) = value.as_slice() {
            self.data.clone_from_slice(s);
            return;
        }
        for (d, off) in self.data.iter_mut().zip(value.layout().iter()) {
            *d = value.data()[off].clone();
        }
    }
}

impl<T: Scalar> Deref for Array<T, 0> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data[0]
    }
}

impl<T: Scalar> DerefMut for Array<T, 0> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data[0]
    }
}

impl<T: Scalar, const N: usize> Index<[usize; N]> for Array<T, N> {
    type Output = T;

    fn index(&self, index: [usize; N]) -> &T {
        &self.data[self.layout.offset(index)]
    }
}

impl<T: Scalar, const N: usize> IndexMut<[usize; N]> for Array<T, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        &mut self.data[self.layout.offset(index)]
    }
}

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Iterates over the scalars in row-major order.
    pub fn iter(&self) -> ArrayIter<'_, T, N> {
        self.view().iter()
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for &'a Array<T, N> {
    type Item = &'a T;
    type IntoIter = ArrayIter<'a, T, N>;

    /// Iterates over the scalars in row-major order.
    fn into_iter(self) -> Self::IntoIter {
        self.view().iter()
    }
}

impl<T: Scalar, const N: usize> IntoIterator for Array<T, N> {
    type Item = T;
    type IntoIter = ArrayIntoIter<T, N>;

    /// Iterates over the scalars in row-major order, consuming the array.
    fn into_iter(self) -> Self::IntoIter {
        ArrayIntoIter {
            inner: self.data.into_iter(),
        }
    }
}

impl<T: Scalar + PartialEq, const N: usize> PartialEq for Array<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.extents() == other.extents() && self.data == other.data
    }
}

impl<T: Scalar + Eq, const N: usize> Eq for Array<T, N> {}

impl<T: Scalar> From<T> for Array<T, 0> {
    fn from(value: T) -> Self {
        Self::scalar(value)
    }
}

impl<T: Scalar> From<Box<[T]>> for Array<T, 1> {
    fn from(value: Box<[T]>) -> Self {
        let len = value.len();
        Self::from_parts([len], value)
    }
}

impl<T: Scalar> From<Vec<T>> for Array<T, 1> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();
        Self::from_parts([len], value.into())
    }
}

impl<T: Scalar, const N0: usize> From<[T; N0]> for Array<T, 1> {
    fn from(value: [T; N0]) -> Self {
        Self::from_parts([N0], value.into())
    }
}

impl<T: Scalar, const N0: usize, const N1: usize> From<[[T; N1]; N0]> for Array<T, 2> {
    fn from(value: [[T; N1]; N0]) -> Self {
        let mut data = Vec::with_capacity(N0 * N1);
        for row in value {
            data.extend_from_slice(&row);
        }
        Self::from_parts([N0, N1], data.into())
    }
}

impl<T: Scalar, const N0: usize, const N1: usize, const N2: usize> From<[[[T; N2]; N1]; N0]>
    for Array<T, 3>
{
    fn from(value: [[[T; N2]; N1]; N0]) -> Self {
        let mut data = Vec::with_capacity(N0 * N1 * N2);
        for matrix in value {
            for row in matrix {
                data.extend_from_slice(&row);
            }
        }
        Self::from_parts([N0, N1, N2], data.into())
    }
}
