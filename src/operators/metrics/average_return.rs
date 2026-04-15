//! Average return since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::time::Instant;
use crate::{Array, Notify, Operator, Scalar};

/// Average (arithmetic mean) of period returns since inception.
pub struct AverageReturn<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> AverageReturn<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

pub struct AverageReturnState<T: Scalar + Float> {
    prev: T,
    sum: T,
    count: usize,
}

impl<T: Scalar + Float> Operator for AverageReturn<T> {
    type State = AverageReturnState<T>;
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, _inputs: (&Array<T>,), _timestamp: Instant) -> (Self::State, Array<T>) {
        (
            AverageReturnState {
                prev: T::nan(),
                sum: T::zero(),
                count: 0,
            },
            Array::scalar(T::nan()),
        )
    }

    fn compute(
        state: &mut AverageReturnState<T>,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let current = inputs.0[0];
        if current.is_nan() {
            return false;
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.count += 1;
            output[0] = state.sum / T::from(state.count).unwrap();
        }

        state.prev = current;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let a = Array::scalar(0.0_f64);
        let (mut s, mut o) = AverageReturn::new().init((&a,), Instant::from_nanos(0));

        let mut a = Array::scalar(100.0);
        AverageReturn::compute(&mut s, (&a,), &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert!(o[0].is_nan()); // no return yet

        a[0] = 110.0;
        AverageReturn::compute(&mut s, (&a,), &mut o, Instant::from_nanos(2), &Notify::new(&[], 0));
        assert!((o[0] - 0.10).abs() < 1e-10);

        a[0] = 99.0;
        AverageReturn::compute(&mut s, (&a,), &mut o, Instant::from_nanos(3), &Notify::new(&[], 0));
        // returns: 0.10, -0.10 => mean = 0.0
        assert!((o[0] - 0.0).abs() < 1e-10);
    }
}
