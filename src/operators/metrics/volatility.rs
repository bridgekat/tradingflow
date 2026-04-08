//! Volatility (population standard deviation of returns) since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::{Array, Notify, Operator, Scalar};

/// Population standard deviation of period returns since inception.
pub struct Volatility<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Volatility<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

pub struct VolatilityState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
}

impl<T: Scalar + Float> Operator for Volatility<T> {
    type State = VolatilityState<T>;
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, _inputs: (&Array<T>,), _timestamp: i64) -> (Self::State, Array<T>) {
        (
            VolatilityState {
                prev: T::nan(),
                sum: T::zero(),
                sum_sq: T::zero(),
                count: 0,
            },
            Array::scalar(T::nan()),
        )
    }

    fn compute(
        state: &mut VolatilityState<T>,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        let current = inputs.0[0];
        if current.is_nan() {
            return false;
        }

        if !state.prev.is_nan() && state.prev > T::zero() {
            let r = current / state.prev - T::one();
            state.sum = state.sum + r;
            state.sum_sq = state.sum_sq + r * r;
            state.count += 1;
            let n = T::from(state.count).unwrap();
            let mean = state.sum / n;
            let var = state.sum_sq / n - mean * mean;
            output[0] = if var > T::zero() { var.sqrt() } else { T::zero() };
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
        let (mut s, mut o) = Volatility::new().init((&a,), 0);

        let mut a = Array::scalar(100.0);
        Volatility::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], 0));

        a[0] = 110.0; // r = 0.10
        Volatility::compute(&mut s, (&a,), &mut o, 2, &Notify::new(&[], 0));
        assert_eq!(o[0], 0.0); // single return => zero std

        a[0] = 99.0; // r = -0.10
        Volatility::compute(&mut s, (&a,), &mut o, 3, &Notify::new(&[], 0));
        // returns: 0.10, -0.10 => mean=0, var=0.01, std=0.10
        assert!((o[0] - 0.10).abs() < 1e-10);
    }
}
