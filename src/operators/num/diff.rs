//! Diff operator — element-wise first difference across ticks.

use std::marker::PhantomData;

use num_traits::Float;

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar};

/// Element-wise difference across ticks: emits `input - input_{offset steps ago}`.
///
/// Maintains a ring buffer of the last `offset` input arrays. For the first
/// `offset` ticks the output is all `NaN`. `offset` must be at least `1`.
///
/// Combined with [`Log`](crate::operators::num::Log) upstream this produces
/// log returns: `Log -> Diff`.
pub struct Diff<T: Scalar + Float> {
    offset: usize,
    _marker: PhantomData<T>,
}

impl<T: Scalar + Float> Diff<T> {
    /// Create a new diff operator with lookback `offset` (>= 1).
    pub fn new(offset: usize) -> Self {
        assert!(offset >= 1, "Diff requires offset >= 1");
        Self {
            offset,
            _marker: PhantomData,
        }
    }
}

/// Runtime state: ring buffer of past input arrays.
pub struct DiffState<T: Scalar + Float> {
    buffer: Vec<Vec<T>>,
    head: usize,
    filled: usize,
}

impl<T: Scalar + Float> Operator for Diff<T> {
    type State = DiffState<T>;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (DiffState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let buffer = vec![vec![T::nan(); stride]; self.offset];
        let out = Array::from_vec(shape, vec![T::nan(); stride]);
        (
            DiffState {
                buffer,
                head: 0,
                filled: 0,
            },
            out,
        )
    }

    fn compute(
        state: &mut DiffState<T>,
        inputs: &Array<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let src = inputs.as_slice();
        let dst = output.as_mut_slice();
        let offset = state.buffer.len();

        if state.filled >= offset {
            let prev = &state.buffer[state.head];
            for i in 0..dst.len() {
                dst[i] = src[i] - prev[i];
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
    fn diff_basic() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = Diff::<f64>::new(1).init(&a, Instant::MIN);
        assert!(out[0].is_nan());

        a[0] = 10.0;
        Diff::compute(&mut state, &a, &mut out, ts(1), false);
        assert!(out[0].is_nan());

        a[0] = 25.0;
        Diff::compute(&mut state, &a, &mut out, ts(2), false);
        assert_eq!(out[0], 15.0);

        a[0] = 20.0;
        Diff::compute(&mut state, &a, &mut out, ts(3), false);
        assert_eq!(out[0], -5.0);
    }

    #[test]
    fn diff_offset_two() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = Diff::<f64>::new(2).init(&a, Instant::MIN);

        a[0] = 10.0;
        Diff::compute(&mut state, &a, &mut out, ts(1), false);
        assert!(out[0].is_nan());

        a[0] = 15.0;
        Diff::compute(&mut state, &a, &mut out, ts(2), false);
        assert!(out[0].is_nan());

        a[0] = 25.0;
        Diff::compute(&mut state, &a, &mut out, ts(3), false);
        assert_eq!(out[0], 15.0);

        a[0] = 30.0;
        Diff::compute(&mut state, &a, &mut out, ts(4), false);
        assert_eq!(out[0], 15.0);
    }

    #[test]
    fn diff_vector() {
        let mut a = Array::from_vec(&[2], vec![0.0_f64; 2]);
        let (mut state, mut out) = Diff::<f64>::new(1).init(&a, Instant::MIN);

        a.assign(&[1.0, 2.0]);
        Diff::compute(&mut state, &a, &mut out, ts(1), false);
        assert!(out.as_slice()[0].is_nan());

        a.assign(&[3.0, 5.0]);
        Diff::compute(&mut state, &a, &mut out, ts(2), false);
        assert_eq!(out.as_slice(), &[2.0, 3.0]);
    }

    #[test]
    fn diff_nan_input_propagates() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = Diff::<f64>::new(1).init(&a, Instant::MIN);

        a[0] = f64::NAN;
        Diff::compute(&mut state, &a, &mut out, ts(1), false);
        assert!(out[0].is_nan());

        a[0] = 10.0;
        Diff::compute(&mut state, &a, &mut out, ts(2), false);
        assert!(out[0].is_nan()); // 10 - NaN = NaN

        a[0] = 15.0;
        Diff::compute(&mut state, &a, &mut out, ts(3), false);
        assert_eq!(out[0], 5.0);
    }
}
