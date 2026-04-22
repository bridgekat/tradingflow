//! Percentage-change operator — element-wise linear returns across ticks.

use std::marker::PhantomData;

use num_traits::Float;

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar};

/// Element-wise linear return across ticks: emits `input / input_{offset steps ago} - 1`.
///
/// Maintains a ring buffer of the last `offset` input arrays. For the first
/// `offset` ticks the output is all `NaN`. `offset` must be at least `1`.
///
/// This is the linear-return counterpart of [`Diff`](crate::operators::num::Diff):
/// `PctChange` yields `p_t / p_{t-k} - 1`, while `Log -> Diff` yields
/// `log p_t - log p_{t-k}`.
pub struct PctChange<T: Scalar + Float> {
    offset: usize,
    _marker: PhantomData<T>,
}

impl<T: Scalar + Float> PctChange<T> {
    /// Create a new percentage-change operator with lookback `offset` (>= 1).
    pub fn new(offset: usize) -> Self {
        assert!(offset >= 1, "PctChange requires offset >= 1");
        Self {
            offset,
            _marker: PhantomData,
        }
    }
}

/// Runtime state: ring buffer of past input arrays.
pub struct PctChangeState<T: Scalar + Float> {
    buffer: Vec<Vec<T>>,
    head: usize,
    filled: usize,
}

impl<T: Scalar + Float> Operator for PctChange<T> {
    type State = PctChangeState<T>;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (PctChangeState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let buffer = vec![vec![T::nan(); stride]; self.offset];
        let out = Array::from_vec(shape, vec![T::nan(); stride]);
        (
            PctChangeState {
                buffer,
                head: 0,
                filled: 0,
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
        let offset = state.buffer.len();
        let one = T::one();

        if state.filled >= offset {
            let prev = &state.buffer[state.head];
            for i in 0..dst.len() {
                dst[i] = src[i] / prev[i] - one;
            }
        } else {
            dst.fill(T::nan());
            state.filled += 1;
        }

        // Overwrite the oldest slot with the new input and advance head.
        state.buffer[state.head].copy_from_slice(src);
        state.head = (state.head + 1) % offset;

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
        let (mut state, mut out) = PctChange::<f64>::new(1).init(&a, Instant::MIN);
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
    fn pct_change_offset_two() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = PctChange::<f64>::new(2).init(&a, Instant::MIN);

        a[0] = 100.0;
        PctChange::compute(&mut state, &a, &mut out, ts(1), false);
        assert!(out[0].is_nan());

        a[0] = 105.0;
        PctChange::compute(&mut state, &a, &mut out, ts(2), false);
        assert!(out[0].is_nan());

        a[0] = 120.0;
        PctChange::compute(&mut state, &a, &mut out, ts(3), false);
        assert!((out[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn pct_change_vector() {
        let mut a = Array::from_vec(&[2], vec![0.0_f64; 2]);
        let (mut state, mut out) = PctChange::<f64>::new(1).init(&a, Instant::MIN);

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
        let (mut state, mut out) = PctChange::<f64>::new(1).init(&a, Instant::MIN);

        a[0] = 0.0;
        PctChange::compute(&mut state, &a, &mut out, ts(1), false);

        a[0] = 1.0;
        PctChange::compute(&mut state, &a, &mut out, ts(2), false);
        assert!(out[0].is_infinite());
    }
}
