//! Inherent impls and conversions for the borrowed [`ArrayView<'a, T, N>`](ArrayView):
//! construction, dimensions, bulk access, per-axis indexing, and materialization.

use std::borrow::Cow;
use std::ops;

use super::{Array, ArrayView};
use crate::{Scalar, Shape};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// View extents and a flat row-major buffer as a contiguous array — the
    /// borrowing counterpart of [`Array::from_vec`].
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != extents.iter().product()`.
    pub fn from_slice(extents: [usize; N], data: &'a [T]) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(
            data.len(),
            shape.len(),
            "from_slice: extents {:?} expect {} scalars, got {}",
            extents,
            shape.len(),
            data.len(),
        );
        Self { data, shape }
    }

    /// Build a strided view from a [`Shape`] and a backing slice whose **first
    /// element is the view's origin** (`[0, …, 0]`).
    ///
    /// # Panics
    ///
    /// Panics if `data` is too short to contain every scalar the shape
    /// addresses (`data.len() < shape.span()`).
    pub fn from_parts(shape: Shape<N>, data: &'a [T]) -> Self {
        assert!(
            data.len() >= shape.span(),
            "from_parts: shape spans {} scalars, got {}",
            shape.span(),
            data.len(),
        );
        Self { data, shape }
    }
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> ArrayView<'_, T, N> {
    /// The shape (per-axis extents and strides; possibly non-canonical).
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// Per-axis extents.
    pub fn extents(&self) -> [usize; N] {
        self.shape.extents()
    }

    /// Number of scalars (product of extents).
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    /// Whether there are no scalars (some extent is zero).
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Bulk access
// ---------------------------------------------------------------------------

impl<'a, T: Scalar, const N: usize> ArrayView<'a, T, N> {
    /// The backing slice from the view's origin — for callers that walk the
    /// view with explicit stride arithmetic (index `0` is the `[0, …, 0]`
    /// element).
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// The contiguous fast path: `Some(flat slice)` iff the view has canonical
    /// row-major strides. `None` for a strided view (e.g. a column).
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.shape.is_contiguous() {
            Some(&self.data[..self.shape.len()])
        } else {
            None
        }
    }

    /// Borrow the view's scalars as a contiguous flat slice, materializing into
    /// an owned buffer (row-major) only when the view is strided. Zero-copy for
    /// the common contiguous case.
    pub fn to_contiguous(&self) -> Cow<'a, [T]> {
        match self.as_slice() {
            Some(s) => Cow::Borrowed(s),
            None => Cow::Owned(self.to_vec()),
        }
    }

    /// Materialize the view into a fresh row-major `Vec<T>`.
    pub fn to_vec(&self) -> Vec<T> {
        match self.as_slice() {
            Some(s) => s.to_vec(),
            None => self
                .shape
                .offsets()
                .map(|off| self.data[off].clone())
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

/// Index by a per-axis logical index, resolved through the view's strides —
/// `v[[i, j]]` is the same scalar as in the parent array, with no copy.
///
/// # Panics
///
/// Panics if the index is out of bounds on any axis.
impl<T: Scalar, const N: usize> ops::Index<[usize; N]> for ArrayView<'_, T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        &self.data[self.shape.offset(index)]
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> ArrayView<'_, T, N> {
    /// Copy the view into an owned, contiguous [`Array`].
    pub fn to_array(&self) -> Array<T, N> {
        Array::from_vec(self.shape.extents(), self.to_vec())
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
