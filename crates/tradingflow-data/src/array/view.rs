use std::borrow::Cow;
use std::ops::{Deref, Index};

use super::{Array, ArrayIter, ArrayOuterIter};
use crate::{Layout, Scalar, layout};

/// A borrowed, strided view of an [`Array`].
#[derive(Debug)]
pub struct ArrayView<'a, T: Scalar, const N: usize> {
    layout: layout::Strided<N>,
    data: &'a [T],
}

impl<'a, T: Scalar> ArrayView<'a, T, 0> {
    /// Creates a rank-0 array view holding one scalar.
    pub fn scalar(value: &'a T) -> Self {
        Self {
            layout: layout::Strided::new([], []),
            data: std::slice::from_ref(value),
        }
    }
}

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// Creates an array view filled with `value`.
    pub fn full(extents: [usize; N], value: &'a T) -> Self {
        Self {
            layout: layout::Strided::new(extents, [0; N]),
            data: std::slice::from_ref(value),
        }
    }

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

    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    pub fn extents(&self) -> [usize; N] {
        self.layout.extents()
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// A view of the sub-region selected by `slices`.
    ///
    /// # Panics
    ///
    /// Panics if `slices` is out of bounds on any axis.
    pub fn slice(&self, slices: impl layout::IntoSlices<N>) -> Self {
        let (offset, layout) = self.layout.slice(slices);
        // SAFETY: `self.data.len() >= self.layout.span() >= offset + layout.span()`.
        unsafe { ArrayView::from_parts_unchecked(layout, &self.data[offset..]) }
    }

    /// A view of the rank-`M` sub-region selected by `slices`.
    ///
    /// # Panics
    ///
    /// Panics if `slices` does not consume exactly `N` axes or produce
    /// exactly `M` axes, or is out of bounds on any axis.
    pub fn slice_reshape<const M: usize, const K: usize>(
        &self,
        slices: impl layout::IntoSliceReshapes<K>,
    ) -> ArrayView<'a, T, M> {
        let (offset, layout) = self.layout.slice_reshape(slices);
        // SAFETY: `self.data.len() >= self.layout.span() >= offset + layout.span()`.
        unsafe { ArrayView::from_parts_unchecked(layout, &self.data[offset..]) }
    }

    /// A view of rank-`M` with `M - N` leading new axes.
    ///
    /// # Panics
    ///
    /// Panics if `M < N`.
    pub fn pad_ndim<const M: usize>(&self) -> ArrayView<'a, T, M> {
        // SAFETY: `self.layout.pad_ndim().span() <= self.layout.span()`.
        unsafe { ArrayView::from_parts_unchecked(self.layout.pad_ndim(), self.data) }
    }

    /// A view with axes swapped: axis `c` and `d` of the result are axes `d`
    /// and `c` of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `c` or `d` is out of bounds for rank `N`.
    pub fn swap_axes(&self, c: usize, d: usize) -> Self {
        let layout = self.layout.swap_axes(c, d);
        // SAFETY: `self.data.len() >= self.layout.span() == layout.span()`.
        unsafe { ArrayView::from_parts_unchecked(layout, self.data) }
    }

    /// A view with axes permuted: axis `d` of the result is axis `perm[d]` of
    /// `self`.
    ///
    /// # Panics
    ///
    /// Panics if `perm` is not a permutation of `0..N`.
    pub fn permute_axes(&self, perm: [usize; N]) -> Self {
        let layout = self.layout.permute_axes(perm);
        // SAFETY: `self.data.len() >= self.layout.span() == layout.span()`.
        unsafe { ArrayView::from_parts_unchecked(layout, self.data) }
    }

    /// A view with axis `from` moved to position `to`, the axes between them
    /// shifting over to keep their relative order.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` is out of bounds for rank `N`.
    pub fn move_axis(&self, from: usize, to: usize) -> Self {
        let layout = self.layout.move_axis(from, to);
        // SAFETY: `self.data.len() >= self.layout.span() == layout.span()`.
        unsafe { ArrayView::from_parts_unchecked(layout, self.data) }
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
            for i in self.layout.iter() {
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

impl<'a, T: Scalar> Deref for ArrayView<'a, T, 0> {
    type Target = T;

    fn deref(&self) -> &'a Self::Target {
        &self.data[0]
    }
}

impl<'a, T: Scalar, const N: usize> Index<[usize; N]> for ArrayView<'a, T, N> {
    type Output = T;

    fn index(&self, index: [usize; N]) -> &'a T {
        &self.data[self.layout.offset(index)]
    }
}

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// Iterates over the scalars in row-major order.
    pub fn iter(&self) -> ArrayIter<'a, T, N> {
        ArrayIter {
            offsets: self.layout().iter(),
            data: self.data(),
        }
    }

    /// Iterates over rank-`N - K` sub-regions in row-major order of the first
    /// `K` axes.
    ///
    /// # Panics
    ///
    /// Panics if `K + M != N`.
    pub fn split_iter<const K: usize, const M: usize>(&self) -> ArrayOuterIter<'a, T, K, M> {
        let (offsets, layout) = self.layout().split_iter();
        ArrayOuterIter {
            offsets,
            layout,
            data: self.data(),
        }
    }
}

impl<'a, T: Scalar, const N: usize> IntoIterator for ArrayView<'a, T, N> {
    type Item = &'a T;
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

impl<T: Scalar + PartialEq, const N: usize> PartialEq for ArrayView<'_, T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.extents() == other.extents() && self.iter().eq(other.iter())
    }
}

impl<T: Scalar + Eq, const N: usize> Eq for ArrayView<'_, T, N> {}
