//! Dense array with dynamic shape, guaranteed standard (C-contiguous) layout.

use std::ops;

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

/// A permitted array scalar type.
pub trait Scalar: Sized + Send + Sync + Clone + Default + 'static {}

impl Scalar for () {}
impl Scalar for bool {}
impl Scalar for i8 {}
impl Scalar for i16 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for u8 {}
impl Scalar for u16 {}
impl Scalar for u32 {}
impl Scalar for u64 {}
impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for String {}

/// An N-dimensional array in standard (row-major contiguous) layout.
///
/// Backed by a flat boxed slice.  Size is fixed after construction (use
/// [`reshape`](Array::reshape) to change shape).  All slice accessors are
/// zero-cost.
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

    /// Create an array from a function of flat index.
    pub fn from_fn(shape: &[usize], f: impl FnMut(usize) -> T) -> Self {
        let len = shape.iter().product::<usize>();
        let data = (0..len).map(f).collect();
        Self {
            data,
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
}

// ===========================================================================
// Raw access (zero-cost)
// ===========================================================================

impl<T: Scalar> Array<T> {
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
}

// ===========================================================================
// Iteration
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// Iterate over immutable element references.
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Iterate over mutable element references.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

// ===========================================================================
// Element-wise transforms
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// Fill all elements with `value`.
    pub fn fill(&mut self, value: T) {
        for x in &mut self.data {
            *x = value.clone();
        }
    }

    /// Apply `f` element-wise, returning a new array with the same shape.
    pub fn map<U: Scalar>(&self, f: impl FnMut(&T) -> U) -> Array<U> {
        Array {
            data: self.data.iter().map(f).collect(),
            shape: self.shape.clone(),
        }
    }

    /// Modify each element in place.
    pub fn map_inplace(&mut self, mut f: impl FnMut(&mut T)) {
        for x in &mut self.data {
            f(x);
        }
    }

    /// Pairwise combine with another same-shaped array.
    ///
    /// # Panics
    ///
    /// Panics if shapes differ.
    pub fn zip_with<U: Scalar>(&self, other: &Self, mut f: impl FnMut(&T, &T) -> U) -> Array<U> {
        assert_eq!(
            self.shape, other.shape,
            "zip_with: shape mismatch {:?} vs {:?}",
            self.shape, other.shape,
        );
        Array {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| f(a, b))
                .collect(),
            shape: self.shape.clone(),
        }
    }
}

// ===========================================================================
// Assignment
// ===========================================================================

impl<T: Scalar> Array<T> {
    /// Copy data from another array (same length).
    #[inline(always)]
    pub fn assign(&mut self, other: &Self) {
        self.data.clone_from_slice(&other.data);
    }

    /// Consume and return the underlying data buffer.
    pub fn into_vec(self) -> Vec<T> {
        self.data.into()
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
}

// ===========================================================================
// Ndarray interop
// ===========================================================================

impl<T: Scalar> Array<T> {
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
// Arithmetic operators (element-wise, allocating)
// ===========================================================================

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident) => {
        // &Array op &Array
        impl<T: Scalar + ops::$trait<Output = T>> ops::$trait for &Array<T> {
            type Output = Array<T>;
            fn $method(self, rhs: &Array<T>) -> Array<T> {
                self.zip_with(rhs, |a, b| a.clone().$method(b.clone()))
            }
        }

        // &Array op scalar
        impl<T: Scalar + ops::$trait<Output = T>> ops::$trait<T> for &Array<T> {
            type Output = Array<T>;
            fn $method(self, rhs: T) -> Array<T> {
                self.map(|a| a.clone().$method(rhs.clone()))
            }
        }

        // Array op Array (owned, forwards to &ref)
        impl<T: Scalar + ops::$trait<Output = T>> ops::$trait for Array<T> {
            type Output = Array<T>;
            fn $method(self, rhs: Array<T>) -> Array<T> {
                (&self).$method(&rhs)
            }
        }

        // Array op scalar (owned, forwards to &ref)
        impl<T: Scalar + ops::$trait<Output = T>> ops::$trait<T> for Array<T> {
            type Output = Array<T>;
            fn $method(self, rhs: T) -> Array<T> {
                (&self).$method(rhs)
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);

impl<T: Scalar + ops::Neg<Output = T>> ops::Neg for &Array<T> {
    type Output = Array<T>;
    fn neg(self) -> Array<T> {
        self.map(|a| -a.clone())
    }
}

impl<T: Scalar + ops::Neg<Output = T>> ops::Neg for Array<T> {
    type Output = Array<T>;
    fn neg(self) -> Array<T> {
        -&self
    }
}

// ===========================================================================
// Standard traits
// ===========================================================================

impl<T: Scalar> Clone for Array<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }
}

impl<T: Scalar + PartialEq> PartialEq for Array<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<T: Scalar + std::fmt::Debug> std::fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Array({:?}, shape={:?})", &self.data, &*self.shape)
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
        assert_eq!(a.len(), 6);
        assert_eq!(a.as_slice(), &[1.0; 6]);

        let b = Array::<f64>::zeros(&[4]);
        assert_eq!(b.as_slice(), &[0.0; 4]);
    }

    #[test]
    fn scalar() {
        let a = Array::scalar(42.0_f64);
        assert_eq!(a.shape(), &[] as &[usize]);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0], 42.0);
    }

    #[test]
    fn from_vec_and_from_fn() {
        let a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0]);

        let b = Array::from_fn(&[4], |i| i as f64 * 10.0);
        assert_eq!(b.as_slice(), &[0.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn assign() {
        let mut a = Array::<f64>::zeros(&[3]);
        let b = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        a.assign(&b);
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
        assert_eq!(a.len(), 6);
    }

    #[test]
    #[should_panic(expected = "reshape")]
    fn reshape_wrong_size() {
        let mut a = Array::<f64>::zeros(&[6]);
        a.reshape(&[2, 2]);
    }

    #[test]
    fn iter_and_iter_mut() {
        let a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let sum: f64 = a.iter().sum();
        assert_eq!(sum, 6.0);

        let mut b = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        b.iter_mut().for_each(|x| *x *= 2.0);
        assert_eq!(b.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn fill_and_map() {
        let mut a = Array::<f64>::zeros(&[3]);
        a.fill(5.0);
        assert_eq!(a.as_slice(), &[5.0, 5.0, 5.0]);

        let b = a.map(|x| x * 2.0);
        assert_eq!(b.as_slice(), &[10.0, 10.0, 10.0]);
    }

    #[test]
    fn map_inplace_and_zip_with() {
        let mut a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        a.map_inplace(|x| *x += 10.0);
        assert_eq!(a.as_slice(), &[11.0, 12.0, 13.0]);

        let b = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        let c = a.zip_with(&b, |x, y| x + y);
        assert_eq!(c.as_slice(), &[21.0, 32.0, 43.0]);
    }

    #[test]
    fn into_vec() {
        let a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let v = a.into_vec();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn arithmetic_ops() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let b = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);

        assert_eq!((&a + &b).as_slice(), &[11.0, 22.0, 33.0]);
        assert_eq!((&b - &a).as_slice(), &[9.0, 18.0, 27.0]);
        assert_eq!((&a * &b).as_slice(), &[10.0, 40.0, 90.0]);
        assert_eq!((&b / &a).as_slice(), &[10.0, 10.0, 10.0]);
        assert_eq!((-&a).as_slice(), &[-1.0, -2.0, -3.0]);
    }

    #[test]
    fn arithmetic_scalar_broadcast() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        assert_eq!((&a + 10.0).as_slice(), &[11.0, 12.0, 13.0]);
        assert_eq!((&a * 3.0).as_slice(), &[3.0, 6.0, 9.0]);
    }

    #[test]
    fn arithmetic_owned() {
        let a = Array::from_vec(&[2], vec![1.0_f64, 2.0]);
        let b = Array::from_vec(&[2], vec![3.0, 4.0]);
        let c = a + b;
        assert_eq!(c.as_slice(), &[4.0, 6.0]);
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

    #[test]
    fn debug_format() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let s = format!("{:?}", a);
        assert!(s.contains("Array("));
        assert!(s.contains("shape="));
    }
}
