//! Percentage-change operator — element-wise linear returns across ticks.

use std::marker::PhantomData;

use num_traits::Float;

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar};

/// Element-wise one-step linear return: emits `input / input_prev - 1`.
///
/// Maintains the previous input array.  The output is all `NaN` on the first
/// tick since no previous value is available.
///
/// This is the linear-return counterpart of [`Diff`](crate::operators::num::Diff):
/// `PctChange` yields `p_t / p_{t-1} - 1`, while `Log -> Diff` yields
/// `log p_t - log p_{t-1}`.
pub struct PctChange<T: Scalar + Float>(PhantomData<T>);

impl<T: Scalar + Float> PctChange<T> {
    /// Create a new percentage-change operator.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float> Default for PctChange<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state: the previous input array, initialised to `NaN` so the
/// first tick's output naturally falls out to `NaN` via NaN propagation.
pub struct PctChangeState<T: Scalar + Float> {
    prev: Vec<T>,
}

impl<T: Scalar + Float> Operator for PctChange<T> {
    type State = PctChangeState<T>;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (PctChangeState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let out = Array::from_vec(shape, vec![T::nan(); stride]);
        (
            PctChangeState {
                prev: vec![T::nan(); stride],
            },
            out,
        )
    }

    fn compute(
        state: &mut PctChangeState<T>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        let one = T::one();

        for i in 0..dst.len() {
            dst[i] = src[i] / state.prev[i] - one;
        }

        state.prev.copy_from_slice(src);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    #[test]
    fn pct_change_basic() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = PctChange::<f64>::new().init(&a, Instant::MIN);
        assert!(out[0].is_nan());

        a[0] = 100.0;
        PctChange::compute(&mut state, &a, &mut out, ts(1), false);
        assert!(out[0].is_nan());

        a[0] = 110.0;
        PctChange::compute(&mut state, &a, &mut out, ts(2), false);
        assert!((out[0] - 0.1).abs() < 1e-12);

        a[0] = 99.0;
        PctChange::compute(&mut state, &a, &mut out, ts(3), false);
        assert!((out[0] - (-0.1)).abs() < 1e-12);
    }

    #[test]
    fn pct_change_vector() {
        let mut a = Array::from_vec(&[2], vec![0.0_f64; 2]);
        let (mut state, mut out) = PctChange::<f64>::new().init(&a, Instant::MIN);

        a.assign(&[10.0, 20.0]);
        PctChange::compute(&mut state, &a, &mut out, ts(1), false);
        assert!(out.as_slice()[0].is_nan());

        a.assign(&[11.0, 25.0]);
        PctChange::compute(&mut state, &a, &mut out, ts(2), false);
        assert!((out.as_slice()[0] - 0.1).abs() < 1e-12);
        assert!((out.as_slice()[1] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn pct_change_zero_denominator_is_inf() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = PctChange::<f64>::new().init(&a, Instant::MIN);

        a[0] = 0.0;
        PctChange::compute(&mut state, &a, &mut out, ts(1), false);

        a[0] = 1.0;
        PctChange::compute(&mut state, &a, &mut out, ts(2), false);
        assert!(out[0].is_infinite());
    }
}
