//! Element-wise power operator.

use num_traits::Float;

use crate::operator::Notify;
use crate::{Array, Operator, Scalar};

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
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: i64) -> (T, Array<T>) {
        (self.n, Array::zeros(inputs.0.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut T,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        let n = *state;
        let a = inputs.0.as_slice();
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
        let (mut s, mut o) = Pow::new(2.0).init((&a,), i64::MIN);
        Pow::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], &[]));
        assert_eq!(o.as_slice(), &[1.0, 4.0, 9.0]);
    }
}
