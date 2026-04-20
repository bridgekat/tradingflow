//! Dense array with dynamic shape, guaranteed standard (C-contiguous) layout.

use std::ops;

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

use super::Scalar;

/// An N-dimensional array in row-major contiguous layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array<T: Scalar> {
    data: Box<[T]>,
    shape: Box<[usize]>,
}

// ===========================================================================
// Construction
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// Create a scalar (0-dimensional) array holding one element.
    pub fn scalar(value: T) -> Self {
        Self {
            data: vec![value].into(),
            shape: Box::new([]),
        }
    }

    /// Create an array filled with `value`.
    pub fn full(shape: &[usize], value: T) -> Self {
        let len = shape.iter().product::<usize>();
        Self {
            data: vec![value; len].into(),
            shape: shape.into(),
        }
    }

    /// Create an array filled with `T::default()` (0 for numeric types).
    pub fn zeros(shape: &[usize]) -> Self {
        Self::full(shape, T::default())
    }

    /// Create an array from a flat buffer and shape.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != shape.iter().product()`.
    pub fn from_vec(shape: &[usize], data: Vec<T>) -> Self {
        let expected = shape.iter().product::<usize>();
        assert_eq!(
            data.len(),
            expected,
            "from_vec: shape {:?} expects {} elements, got {}",
            shape,
            expected,
            data.len(),
        );
        Self {
            data: data.into(),
            shape: shape.into(),
        }
    }
}

// ===========================================================================
// Dimensions
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// The array shape.
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    #[inline(always)]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of scalars (product of shape dimensions).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.data.len()
    }
}

// ===========================================================================
// Bulk access
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// Flat immutable slice of all elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Flat mutable slice of all elements.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// ===========================================================================
// Indexing
// ===========================================================================

impl<T: Scalar> ops::Index<usize> for Array<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: Scalar> ops::IndexMut<usize> for Array<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

// ===========================================================================
// Ndarray views
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// Immutable ndarray view (zero-copy).
    pub fn view(&self) -> ArrayViewD<'_, T> {
        ArrayViewD::from_shape(IxDyn(&self.shape), &self.data).unwrap()
    }

    /// Mutable ndarray view (zero-copy).
    pub fn view_mut(&mut self) -> ArrayViewMutD<'_, T> {
        ArrayViewMutD::from_shape(IxDyn(&self.shape), &mut self.data).unwrap()
    }
}

// ===========================================================================
// Assignment
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// Copy data from a slice.
    ///
    /// # Panics
    ///
    /// Panics if `value.len() != self.stride()`.
    #[inline(always)]
    pub fn assign(&mut self, value: &[T]) {
        assert_eq!(
            value.len(),
            self.stride(),
            "push: expected {} scalars, got {}",
            self.stride(),
            value.len(),
        );
        self.data.clone_from_slice(value);
    }
}

// ===========================================================================
// Reshape
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// Change the shape without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different stride.
    pub fn reshape(&mut self, new_shape: &[usize]) {
        let new_stride = new_shape.iter().product::<usize>();
        assert_eq!(
            self.stride(),
            new_stride,
            "reshape: current stride {} != new shape stride {}",
            self.stride(),
            new_stride,
        );
        self.shape = new_shape.into();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_zeros() {
        let a = Array::full(&[2, 3], 1.0_f64);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.stride(), 6);
        assert_eq!(a.as_slice(), &[1.0; 6]);

        let b = Array::<f64>::zeros(&[4]);
        assert_eq!(b.as_slice(), &[0.0; 4]);
    }

    #[test]
    fn scalar() {
        let a = Array::scalar(42.0_f64);
        assert_eq!(a.shape(), &[] as &[usize]);
        assert_eq!(a.stride(), 1);
        assert_eq!(a[0], 42.0);
    }

    #[test]
    fn assign() {
        let mut a = Array::<f64>::zeros(&[3]);
        let b = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        a.assign(b.as_slice());
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn index_mut() {
        let mut a = Array::<f64>::zeros(&[2]);
        a[0] = 10.0;
        a[1] = 20.0;
        assert_eq!(a.as_slice(), &[10.0, 20.0]);
    }

    #[test]
    fn ndarray_view() {
        let a = Array::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = a.view();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v[[0, 0]], 1.0);
        assert_eq!(v[[1, 2]], 6.0);
    }

    #[test]
    fn reshape() {
        let mut a = Array::from_vec(&[6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.reshape(&[2, 3]);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.stride(), 6);
    }

    #[test]
    #[should_panic(expected = "reshape")]
    fn reshape_wrong_size() {
        let mut a = Array::<f64>::zeros(&[6]);
        a.reshape(&[2, 2]);
    }

    #[test]
    fn partial_eq() {
        let a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let c = Array::from_vec(&[3], vec![1.0, 2.0, 4.0]);
        let d = Array::from_vec(&[1, 3], vec![1.0, 2.0, 3.0]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d); // different shape
    }
}
