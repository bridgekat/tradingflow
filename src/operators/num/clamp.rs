//! Element-wise clamp operator.

use num_traits::Float;

use crate::Instant;
use crate::{Array, Input, Notify, Operator, Scalar};

/// Element-wise clamp to `[lo, hi]`.
pub struct Clamp<T: Scalar> {
    lo: T,
    hi: T,
}

impl<T: Scalar + Float> Clamp<T> {
    /// Create a new instance clamping to `[lo, hi]`.
    pub fn new(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }
}

impl<T: Scalar + Float> Operator for Clamp<T> {
    type State = (T, T);
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> ((T, T), Array<T>) {
        ((self.lo, self.hi), Array::zeros(inputs.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut (T, T),
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let (lo, hi) = *state;
        let a = inputs.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = lo.max(hi.min(a[i]));
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp() {
        let a = Array::from_vec(&[3], vec![1.0_f64, 3.0, 7.0]);
        let (mut s, mut o) = Clamp::new(2.0, 5.0).init(&a, Instant::MIN);
        Clamp::compute(&mut s, &a, &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(o.as_slice(), &[2.0, 3.0, 5.0]);
    }
}
