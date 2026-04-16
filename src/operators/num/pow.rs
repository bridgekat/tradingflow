//! Element-wise power operator.

use num_traits::Float;

use crate::Instant;
use crate::{Array, Input, Notify, Operator, Scalar};

/// Element-wise power: `x.powf(n)`.
pub struct Pow<T: Scalar> {
    n: T,
}

impl<T: Scalar + Float> Pow<T> {
    /// Create a new instance with exponent `n`.
    pub fn new(n: T) -> Self {
        Self { n }
    }
}

impl<T: Scalar + Float> Operator for Pow<T> {
    type State = T;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (T, Array<T>) {
        (self.n, Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut T,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let n = *state;
        let a = inputs.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = a[i].powf(n);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 2.0, 3.0]);
        let (mut s, mut o) = Pow::new(2.0).init(&a, Instant::MIN);
        Pow::compute(&mut s, &a, &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[1.0, 4.0, 9.0]);
    }
}
