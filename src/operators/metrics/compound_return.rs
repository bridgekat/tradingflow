//! Compound return since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::time::Instant;
use crate::{Array, Notify, Operator, Scalar};

/// Compound return: `(current / first)^(1/n) - 1` where `n` is the
/// number of periods elapsed.
pub struct CompoundReturn<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> CompoundReturn<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

pub struct CompoundReturnState<T: Scalar + Float> {
    first_value: T,
    count: usize,
}

impl<T: Scalar + Float> Operator for CompoundReturn<T> {
    type State = CompoundReturnState<T>;
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, _inputs: (&Array<T>,), _timestamp: Instant) -> (Self::State, Array<T>) {
        (
            CompoundReturnState {
                first_value: T::nan(),
                count: 0,
            },
            Array::scalar(T::nan()),
        )
    }

    fn compute(
        state: &mut CompoundReturnState<T>,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let current = inputs.0[0];
        if current.is_nan() {
            return false;
        }

        state.count += 1;

        if state.first_value.is_nan() {
            state.first_value = current;
            output[0] = T::zero();
            return true;
        }

        if state.first_value <= T::zero() || current <= T::zero() {
            output[0] = T::nan();
            return true;
        }

        let ratio = current / state.first_value;
        let n = T::from(state.count - 1).unwrap();
        if n <= T::zero() {
            output[0] = T::zero();
        } else {
            output[0] = ratio.powf(T::one() / n) - T::one();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let a = Array::scalar(0.0_f64);
        let (mut s, mut o) = CompoundReturn::new().init((&a,), Instant::from_nanos(0));

        let mut a = Array::scalar(100.0);
        CompoundReturn::compute(&mut s, (&a,), &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(o[0], 0.0); // first tick

        a[0] = 110.0;
        CompoundReturn::compute(&mut s, (&a,), &mut o, Instant::from_nanos(2), &Notify::new(&[], 0));
        assert!((o[0] - 0.10).abs() < 1e-10); // 1 period: 10%

        a[0] = 121.0;
        CompoundReturn::compute(&mut s, (&a,), &mut o, Instant::from_nanos(3), &Notify::new(&[], 0));
        // 2 periods: (121/100)^(1/2) - 1 = 0.10
        assert!((o[0] - 0.10).abs() < 1e-10);
    }
}
