use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

/// An observable *n*-dimensional array value.
///
/// An [`Observable<T>`] holds a fixed-size buffer of `stride` scalar values
/// where `stride` is the product of the element shape dimensions.
/// The buffer is allocated once and overwritten on each update.
/// Timestamps are *not* stored — the scenario passes them externally.
///
/// Every observable is created with an explicit initial value.
/// Missing data should be represented via sentinel values (0, NaN, etc.).
#[derive(Debug)]
pub struct Observable<T: Copy> {
    value: Box<[T]>,
    shape: Box<[usize]>,
}

impl<T: Copy> Observable<T> {
    /// Create an observable with an explicit initial value.
    ///
    /// `value.len()` must equal the stride (product of shape dimensions).
    pub fn new(shape: &[usize], value: &[T]) -> Self {
        assert!(value.len() == shape.iter().product::<usize>());
        Self {
            value: value.into(),
            shape: shape.into(),
        }
    }

    /// Element shape (e.g. `&[2, 3]` for a 2x3 matrix element).
    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of values per element (product of shape dimensions).
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.value.len()
    }

    /// The current value as a slice of length `stride`.
    #[inline(always)]
    pub fn current(&self) -> &[T] {
        &self.value
    }

    /// The current value as a mutable slice of length `stride`.
    #[inline(always)]
    pub fn current_mut(&mut self) -> &mut [T] {
        &mut self.value
    }

    /// Array view of the current value with the element shape.
    #[inline(always)]
    pub fn current_view(&self) -> ArrayViewD<'_, T> {
        // SAFETY: the shape is valid for the buffer which has length `stride`.
        unsafe { ArrayViewD::from_shape_ptr(IxDyn(&self.shape), self.value.as_ptr()) }
    }

    /// Mutable array view of the current value with the element shape.
    #[inline(always)]
    pub fn current_view_mut(&mut self) -> ArrayViewMutD<'_, T> {
        // SAFETY: the shape is valid for the buffer which has length `stride`.
        unsafe { ArrayViewMutD::from_shape_ptr(IxDyn(&self.shape), self.value.as_mut_ptr()) }
    }

    /// Overwrite the current value from a slice.
    ///
    /// `value.len()` must equal the stride (product of shape dimensions).
    #[inline(always)]
    pub fn write(&mut self, value: &[T]) {
        self.current_mut().copy_from_slice(value);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_with_initial() {
        let obs = Observable::new(&[3], &[1.0, 2.0, 3.0]);
        assert_eq!(obs.stride(), 3);
        assert_eq!(obs.shape(), &[3]);
        assert_eq!(obs.current_view().as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn write_overwrites() {
        let mut obs = Observable::new(&[], &[0.0]);
        obs.write(&[42.0]);
        assert_eq!(obs.current_view().as_slice().unwrap(), &[42.0]);
        obs.write(&[99.0]);
        assert_eq!(obs.current_view().as_slice().unwrap(), &[99.0]);
    }

    #[test]
    fn vals_mut_in_place() {
        let mut obs = Observable::new(&[2], &[0.0, 0.0]);
        let buf = obs.current_mut();
        buf[0] = 10.0;
        buf[1] = 20.0;
        assert_eq!(obs.current(), &[10.0, 20.0]);
    }

    #[test]
    fn strided_observable() {
        let mut obs = Observable::new(&[2, 3], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(obs.stride(), 6);
        assert_eq!(obs.shape(), &[2, 3]);
        assert_eq!(obs.current(), &[1, 2, 3, 4, 5, 6]);
        obs.write(&[10, 20, 30, 40, 50, 60]);
        assert_eq!(obs.current(), &[10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn array_view_scalar_zero_copy() {
        let obs = Observable::new(&[], &[42.0]);
        let view = obs.current_view();
        // Scalar → 0-dimensional view.
        assert_eq!(view.shape(), &[] as &[usize]);
        assert_eq!(view.ndim(), 0);
        assert_eq!(view[[]], 42.0);
        // Verify zero-copy.
        assert_eq!(view.as_slice().unwrap().as_ptr(), obs.current().as_ptr(),);
    }

    #[test]
    fn array_view_vector_zero_copy() {
        let obs = Observable::new(&[3], &[1.0, 2.0, 3.0]);
        let view = obs.current_view();
        assert_eq!(view.shape(), &[3]);
        assert_eq!(view[[0]], 1.0);
        assert_eq!(view[[2]], 3.0);
        assert_eq!(view.as_slice().unwrap().as_ptr(), obs.current().as_ptr(),);
    }

    #[test]
    fn array_view_matrix_zero_copy() {
        let obs = Observable::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let view = obs.current_view();
        assert_eq!(view.shape(), &[2, 3]);
        // Row 0, col 2
        assert_eq!(view[[0, 2]], 3.0);
        // Row 1, col 0
        assert_eq!(view[[1, 0]], 4.0);
        assert_eq!(view.as_slice().unwrap().as_ptr(), obs.current().as_ptr(),);
    }
}
