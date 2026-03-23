//! Dense array with dynamic shape, guaranteed standard (C-contiguous) layout.
//!
//! Unlike `ndarray::ArrayD`, all slice accessors are O(1) with no runtime
//! contiguity checks — contiguity is an invariant enforced by construction.

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

use super::types::Scalar;

/// A dense, dynamically-shaped array in standard (row-major) layout.
///
/// Backed by a flat `Vec<T>`.  Shape is fixed after construction (use
/// [`reshape`](Array::reshape) to change it).  All slice accessors are
/// zero-cost.
pub struct Array<T: Scalar> {
    data: Vec<T>,
    shape: Box<[usize]>,
}

impl<T: Scalar> Array<T> {
    /// Create an array filled with `value`.
    pub fn from_elem(shape: &[usize], value: T) -> Self {
        let len = shape.iter().product::<usize>();
        Self {
            data: vec![value; len],
            shape: shape.into(),
        }
    }

    /// Create a zero-filled array.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::from_elem(shape, T::default())
    }

    /// Create an array from a flat buffer and shape.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != shape.iter().product()` (or 1 for scalar).
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
            data,
            shape: shape.into(),
        }
    }

    /// Create a scalar (0-dimensional) array.
    pub fn scalar(value: T) -> Self {
        Self {
            data: vec![value],
            shape: Box::new([]),
        }
    }

    // -- Dimensions ----------------------------------------------------------

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

    /// Total number of elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array has zero elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // -- Slice access (zero-cost) --------------------------------------------

    /// Flat immutable slice of all elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Flat mutable slice of all elements.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Raw pointer to the data.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Mutable raw pointer to the data.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    // -- Element access ------------------------------------------------------

    /// Copy data from another array (same length).
    #[inline(always)]
    pub fn assign(&mut self, other: &Self) {
        self.data.clone_from_slice(&other.data);
    }

    /// Copy data from a slice (same length).
    #[inline(always)]
    pub fn assign_slice(&mut self, src: &[T]) {
        self.data.clone_from_slice(src);
    }

    // -- Reshape -------------------------------------------------------------

    /// Change the shape without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different total element count.
    pub fn reshape(&mut self, new_shape: &[usize]) {
        let new_len = new_shape.iter().product::<usize>();
        assert_eq!(
            self.data.len(),
            new_len,
            "reshape: current len {} != new shape {:?} (len {})",
            self.data.len(),
            new_shape,
            new_len,
        );
        self.shape = new_shape.into();
    }

    // -- Ndarray interop -----------------------------------------------------

    /// Immutable ndarray view (zero-copy).
    pub fn view(&self) -> ArrayViewD<'_, T> {
        let shape = if self.shape.is_empty() {
            IxDyn(&[])
        } else {
            IxDyn(&self.shape)
        };
        ArrayViewD::from_shape(shape, &self.data).unwrap()
    }

    /// Mutable ndarray view (zero-copy).
    pub fn view_mut(&mut self) -> ArrayViewMutD<'_, T> {
        let shape = if self.shape.is_empty() {
            IxDyn(&[])
        } else {
            IxDyn(&self.shape)
        };
        ArrayViewMutD::from_shape(shape, &mut self.data).unwrap()
    }
}

impl<T: Scalar> Clone for Array<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }
}

// -- Index by flat offset ----------------------------------------------------

impl<T: Scalar> std::ops::Index<usize> for Array<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: Scalar> std::ops::IndexMut<usize> for Array<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_from_elem() {
        let a = Array::from_elem(&[2, 3], 1.0_f64);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.len(), 6);
        assert_eq!(a.as_slice(), &[1.0; 6]);
    }

    #[test]
    fn array_scalar() {
        let a = Array::scalar(42.0_f64);
        assert_eq!(a.shape(), &[] as &[usize]);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0], 42.0);
    }

    #[test]
    fn array_from_vec() {
        let a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn array_assign() {
        let mut a = Array::zeros(&[3]);
        let b = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        a.assign(&b);
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn array_index_mut() {
        let mut a = Array::zeros(&[2]);
        a[0] = 10.0;
        a[1] = 20.0;
        assert_eq!(a.as_slice(), &[10.0, 20.0]);
    }

    #[test]
    fn array_ndarray_view() {
        let a = Array::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = a.view();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v[[0, 0]], 1.0);
        assert_eq!(v[[1, 2]], 6.0);
    }

    #[test]
    fn array_reshape() {
        let mut a = Array::from_vec(&[6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.reshape(&[2, 3]);
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.len(), 6);
    }

    #[test]
    #[should_panic(expected = "reshape")]
    fn array_reshape_wrong_size() {
        let mut a = Array::<f64>::zeros(&[6]);
        a.reshape(&[2, 2]);
    }
}
