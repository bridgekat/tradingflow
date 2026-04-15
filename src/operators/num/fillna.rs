//! Fill-NaN operator.

use num_traits::Float;

use crate::data::Instant;
use crate::{Array, Input, Notify, Operator, Scalar};

/// Element-wise NaN replacement: replaces each NaN with `val`.
pub struct Fillna<T: Scalar> {
    val: T,
}

impl<T: Scalar + Float> Fillna<T> {
    /// Create a new instance replacing NaN with `val`.
    pub fn new(val: T) -> Self {
        Self { val }
    }
}

impl<T: Scalar + Float> Operator for Fillna<T> {
    type State = T;
    type Inputs = (Input<Array<T>>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: Instant) -> (T, Array<T>) {
        (self.val, Array::zeros(inputs.0.shape()))
    }

    #[inline(always)]
    fn compute(
        state: &mut T,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let val = *state;
        let a = inputs.0.as_slice();
        let out = output.as_mut_slice();
        for i in 0..out.len() {
            out[i] = if a[i].is_nan() { val } else { a[i] };
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fillna() {
        let a = Array::from_vec(&[3], vec![1.0_f64, f64::NAN, 3.0]);
        let (mut s, mut o) = Fillna::new(0.0).init((&a,), Instant::MIN);
        Fillna::compute(&mut s, (&a,), &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(o[0], 1.0);
        assert_eq!(o[1], 0.0);
        assert_eq!(o[2], 3.0);
    }
}
