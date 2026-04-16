//! Element-wise scale operator.

use std::ops;

use crate::Instant;
use crate::{Array, Input, Notify, Operator, Scalar};

/// Element-wise scale: `x * c`.
pub struct Scale<T: Scalar> {
    c: T,
}

impl<T: Scalar + ops::Mul<Output = T>> Scale<T> {
    /// Create a new instance with factor `c`.
    pub fn new(c: T) -> Self {
        Self { c }
    }
}

impl<T: Scalar + ops::Mul<Output = T>> Operator for Scale<T> {
    type State = T;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (T, Array<T>) {
        (self.c, Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut T,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let c = state.clone();
        let a = inputs.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = a[i].clone() * c.clone();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let (mut s, mut o) = Scale::new(3.0).init(&a, Instant::MIN);
        Scale::compute(&mut s, &a, &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[3.0, 6.0, 9.0]);
    }
}
