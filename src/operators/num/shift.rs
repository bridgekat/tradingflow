//! Element-wise shift operator.

use std::ops;

use crate::{Array, Notify, Operator, Scalar};

/// Element-wise shift: `x + c`.
pub struct Shift<T: Scalar> {
    c: T,
}

impl<T: Scalar + ops::Add<Output = T>> Shift<T> {
    /// Create a new instance with offset `c`.
    pub fn new(c: T) -> Self {
        Self { c }
    }
}

impl<T: Scalar + ops::Add<Output = T>> Operator for Shift<T> {
    type State = T;
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: i64) -> (T, Array<T>) {
        (self.c, Array::zeros(inputs.0.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut T,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        let c = state.clone();
        let a = inputs.0.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = a[i].clone() + c.clone();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shift() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let (mut s, mut o) = Shift::new(10.0).init((&a,), i64::MIN);
        Shift::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[11.0, 12.0, 13.0]);
    }
}
