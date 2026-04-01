//! Cast operator — element-wise type conversion between arrays.

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use crate::{Array, Operator, Scalar};

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
    type Inputs = (Array<S>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<S>,), _timestamp: i64) -> ((), Array<T>) {
        let src = inputs.0.as_slice();
        let data: Vec<T> = src.iter().map(|&v| v.as_()).collect();
        ((), Array::from_vec(inputs.0.shape(), data))
    }

    #[inline(always)]
    fn compute(
        _state: &mut (),
        inputs: (&Array<S>,),
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        let src = inputs.0.as_slice();
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
    use crate::operator::Operator;

    #[test]
    fn cast_i32_to_f64() {
        let a = Array::from_vec(&[3], vec![1_i32, 2, 3]);
        let (mut s, mut o) = Cast::<i32, f64>::new().init((&a,), i64::MIN);
        assert_eq!(o.as_slice(), &[1.0, 2.0, 3.0]);

        let b = Array::from_vec(&[3], vec![10_i32, 20, 30]);
        assert!(Cast::<i32, f64>::compute(&mut s, (&b,), &mut o, 1));
        assert_eq!(o.as_slice(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn cast_f64_to_i32() {
        let a = Array::from_vec(&[2], vec![1.9_f64, -2.7]);
        let (_, o) = Cast::<f64, i32>::new().init((&a,), i64::MIN);
        assert_eq!(o.as_slice(), &[1, -2]); // truncation
    }
}
