//! Cast operator — element-wise type conversion between arrays.

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar};

/// Element-wise type conversion: `out[i] = input[i] as T`.
///
/// Uses [`num_traits::AsPrimitive`] for the scalar conversion, which
/// corresponds to Rust's `as` semantics (truncating / saturating).
pub struct Cast<S: Scalar, T: Scalar> {
    _phantom: PhantomData<(S, T)>,
}

impl<S: Scalar, T: Scalar> Cast<S, T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S: Scalar, T: Scalar> Default for Cast<S, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, T> Operator for Cast<S, T>
where
    S: Scalar + Copy + AsPrimitive<T>,
    T: Scalar + Copy + 'static,
{
    type State = ();
    type Inputs = Input<Array<S>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<S>, _timestamp: Instant) -> ((), Array<T>) {
        let src = inputs.as_slice();
        let data: Vec<T> = src.iter().map(|&v| v.as_()).collect();
        ((), Array::from_vec(inputs.shape(), data))
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        inputs: &Array<S>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        for i in 0..dst.len() {
            dst[i] = src[i].as_();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instant;
    use crate::operator::Operator;

    #[test]
    fn cast_i32_to_f64() {
        let a = Array::from_vec(&[3], vec![1_i32, 2, 3]);
        let (mut s, mut o) = Cast::<i32, f64>::new().init(&a, Instant::MIN);
        assert_eq!(o.as_slice(), &[1.0, 2.0, 3.0]);

        let b = Array::from_vec(&[3], vec![10_i32, 20, 30]);
        assert!(Cast::<i32, f64>::compute(
            &mut s,
            &b,
            &mut o,
            Instant::from_nanos(1),
            false
        ));
        assert_eq!(o.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn cast_f64_to_i32() {
        let a = Array::from_vec(&[2], vec![1.9_f64, -2.7]);
        let (_, o) = Cast::<f64, i32>::new().init(&a, Instant::MIN);
        assert_eq!(o.as_slice(), &[1, -2]); // truncation
    }
}
