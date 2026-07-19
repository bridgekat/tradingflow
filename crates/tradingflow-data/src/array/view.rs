use std::borrow::Cow;
use std::ops::Index;

use super::{Array, ArrayIter};
use crate::{Layout, Scalar, layout};

/// A borrowed, strided view of an [`Array`].
#[derive(Debug, PartialEq, Eq)]
pub struct ArrayView<'a, T: Scalar, const N: usize> {
    layout: layout::Strided<N>,
    data: &'a [T],
}

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// Creates an array view from a row-major contiguous slice.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != extents.iter().product()`.
    pub fn from_slice(extents: [usize; N], data: &'a [T]) -> Self {
        let layout = layout::RowMajor::new(extents);
        assert_eq!(
            data.len(),
            layout.len(),
            "from_slice: extents {:?} expect {} scalars, got {}",
            extents,
            layout.len(),
            data.len(),
        );
        Self {
            layout: layout.into(),
            data,
        }
    }

    /// Creates an array view from a row-major contiguous slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data.len() == extents.iter().product()`.
    pub unsafe fn from_slice_unchecked(extents: [usize; N], data: &'a [T]) -> Self {
        let layout = layout::RowMajor::new(extents);
        debug_assert_eq!(data.len(), layout.len());
        Self {
            layout: layout.into(),
            data,
        }
    }

    /// Creates an array view from a strided slice.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < layout.span()`.
    pub fn from_parts(layout: layout::Strided<N>, data: &'a [T]) -> Self {
        assert!(
            data.len() >= layout.span(),
            "from_parts: shape spans {} scalars, got {}",
            layout.span(),
            data.len(),
        );
        Self { layout, data }
    }

    /// Creates an array view from a strided slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data.len() >= layout.span()`.
    pub unsafe fn from_parts_unchecked(layout: layout::Strided<N>, data: &'a [T]) -> Self {
        debug_assert!(data.len() >= layout.span());
        Self { layout, data }
    }

    pub fn layout(&self) -> layout::Strided<N> {
        self.layout
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// Returns `Some(data)` if the view has row-major contiguous layout.
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.layout.is_contiguous() {
            Some(&self.data[..self.layout.len()])
        } else {
            None
        }
    }

    /// Borrows the view's scalars as a row-major contiguous slice,
    /// materializing into an owned buffer if needed.
    pub fn to_contiguous(&self) -> Cow<'a, [T]> {
        if let Some(slice) = self.as_slice() {
            Cow::Borrowed(slice)
        } else {
            let mut owned = Vec::with_capacity(self.layout.len());
            for i in self.layout.offsets() {
                owned.push(self.data[i].clone());
            }
            Cow::Owned(owned)
        }
    }

    /// Copy the view into an owned, contiguous [`Array`].
    pub fn to_array(&self) -> Array<T, N> {
        // SAFETY: `to_contiguous()` returns a slice of length `self.layout.len()`.
        unsafe { Array::from_parts_unchecked(self.layout.extents(), self.to_contiguous().into()) }
    }
}

impl<T: Scalar, const N: usize> Layout<N> for ArrayView<'_, T, N> {
    fn extents(&self) -> [usize; N] {
        self.layout.extents()
    }

    fn strides(&self) -> [usize; N] {
        self.layout.strides()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T: Scalar, const N: usize> Index<[usize; N]> for ArrayView<'_, T, N> {
    type Output = T;

    fn index(&self, index: [usize; N]) -> &T {
        &self.data[self.layout.offset(index)]
    }
}

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// Iterates over the scalars in row-major order.
    pub fn iter(&self) -> ArrayIter<'a, T, N> {
        ArrayIter {
            offsets: self.layout().offsets(),
            data: self.data(),
        }
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for ArrayView<'a, T, N> {
    type Item = T;
    type IntoIter = ArrayIter<'a, T, N>;

    /// Iterates over the scalars in row-major order.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Scalar, const N: usize> From<&'a Array<T, N>> for ArrayView<'a, T, N> {
    fn from(a: &'a Array<T, N>) -> Self {
        a.view()
    }
}

impl<T: Scalar, const N: usize> From<ArrayView<'_, T, N>> for Array<T, N> {
    fn from(v: ArrayView<'_, T, N>) -> Self {
        v.to_array()
    }
}

impl<T: Scalar, const N: usize> Clone for ArrayView<'_, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Scalar, const N: usize> Copy for ArrayView<'_, T, N> {}
