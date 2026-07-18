//! Inherent impls for the owned [`Array<T, N>`](Array): construction,
//! dimensions, bulk and per-axis access, and in-place mutation.

use std::ops;

use super::{Array, ArrayView, write_row_major};
use crate::{Scalar, Shape};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<T: Scalar> Array<T, 0> {
    /// Create a rank-0 array holding one scalar.
    pub fn scalar(value: T) -> Self {
        Self {
            data: vec![value].into(),
            shape: Shape::row_major([]),
        }
    }
}

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Create an array filled with `value`.
    pub fn full(extents: [usize; N], value: T) -> Self {
        let shape = Shape::row_major(extents);
        Self {
            data: vec![value; shape.len()].into(),
            shape,
        }
    }

    /// Create an array filled with `T::default()` (0 for numeric types).
    pub fn zeros(extents: [usize; N]) -> Self {
        Self::full(extents, T::default())
    }

    /// Create an array from extents and a flat row-major buffer — the owning
    /// counterpart of [`ArrayView::from_slice`].
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != extents.iter().product()`.
    pub fn from_vec(extents: [usize; N], data: Vec<T>) -> Self {
        let shape = Shape::row_major(extents);
        assert_eq!(
            data.len(),
            shape.len(),
            "from_vec: extents {:?} expect {} scalars, got {}",
            extents,
            shape.len(),
            data.len(),
        );
        Self {
            data: data.into(),
            shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Array<T, N> {
    /// The shape (per-axis extents and strides; always canonical row-major).
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

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Flat immutable slice of all scalars (row-major).
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Flat mutable slice of all scalars (row-major).
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow the whole array as a contiguous [`ArrayView`].
    pub fn view(&self) -> ArrayView<'_, T, N> {
        ArrayView {
            data: &self.data,
            shape: self.shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

/// Index by a per-axis logical index — `a[[i, j]]` for a rank-2 array, `a[[]]`
/// for a rank-0 one. [`data`](Array::data) is the flat row-major escape hatch.
///
/// # Panics
///
/// Panics if the index is out of bounds on any axis.
impl<T: Scalar, const N: usize> ops::Index<[usize; N]> for Array<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        &self.data[self.shape.offset(index)]
    }
}

impl<T: Scalar, const N: usize> ops::IndexMut<[usize; N]> for Array<T, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        &mut self.data[self.shape.offset(index)]
    }
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

impl<T: Scalar, const N: usize> Array<T, N> {
    /// Copy in the scalars of a rank-`N` view, which may be strided (it is
    /// materialized row-major). [`data_mut`](Self::data_mut) is the flat
    /// counterpart.
    ///
    /// # Panics
    ///
    /// Panics if `value.extents() != self.extents()`.
    pub fn assign(&mut self, value: ArrayView<'_, T, N>) {
        assert_eq!(value.extents(), self.extents(), "assign: extents mismatch");
        write_row_major(&mut self.data, value);
    }

    /// Change the extents in place (same rank), without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new extents have a different scalar count.
    pub fn reshape(&mut self, extents: [usize; N]) {
        let shape = Shape::row_major(extents);
        assert_eq!(
            self.len(),
            shape.len(),
            "reshape: current len {} != new extents {:?} ({} scalars)",
            self.len(),
            extents,
            shape.len(),
        );
        self.shape = shape;
    }
}
