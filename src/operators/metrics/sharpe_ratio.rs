//! Sharpe ratio since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::{Array, Notify, Operator, Scalar};

/// Sharpe ratio: `mean(r) / std(r)` of period returns since inception.
///
/// Returns `NaN` until at least two period returns have been observed
/// (need non-zero variance).
pub struct SharpeRatio<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> SharpeRatio<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

pub struct SharpeRatioState<T: Scalar + Float> {
    prev: T,
    sum: T,
    sum_sq: T,
    count: usize,
}

impl<T: Scalar + Float> Operator for SharpeRatio<T> {
    type State = SharpeRatioState<T>;
    type Inputs = (Array<T>,);
    type Output = Array<T>;

    fn init(self, _inputs: (&Array<T>,), _timestamp: i64) -> (Self::State, Array<T>) {
        (
            SharpeRatioState {
                prev: T::nan(),
                sum: T::zero(),
                sum_sq: T::zero(),
                count: 0,
            },
            Array::scalar(T::nan()),
        )
    }

    fn compute(
        state: &mut SharpeRatioState<T>,
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

            output[0] = if var > T::zero() {
                mean / var.sqrt()
            } else {
                T::nan()
            };
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
        let (mut s, mut o) = SharpeRatio::new().init((&a,), 0);

        let mut a = Array::scalar(100.0);
        SharpeRatio::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], &[]));

        a[0] = 110.0; // r = 0.10
        SharpeRatio::compute(&mut s, (&a,), &mut o, 2, &Notify::new(&[], &[]));
        // single return with zero variance => NaN
        assert!(o[0].is_nan());

        a[0] = 121.0; // r = 0.10
        SharpeRatio::compute(&mut s, (&a,), &mut o, 3, &Notify::new(&[], &[]));
        // returns: 0.10, 0.10 => mean=0.10, std=0 => NaN
        assert!(o[0].is_nan());

        a[0] = 115.0; // r ≈ -0.0496
        SharpeRatio::compute(&mut s, (&a,), &mut o, 4, &Notify::new(&[], &[]));
        // returns: 0.10, 0.10, -0.0496 => mean>0, std>0 => finite
        assert!(o[0].is_finite());
        assert!(o[0] > 0.0); // positive mean, positive Sharpe
    }
}
