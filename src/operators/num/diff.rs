//! Diff operator — element-wise first difference across ticks.

use std::marker::PhantomData;

use num_traits::Float;

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar};

/// Element-wise first difference across ticks: emits `input - input_prev`.
///
/// Maintains the previous input array.  The output is all `NaN` on the first
/// tick since no previous value is available.
///
/// Combined with [`Log`](crate::operators::num::Log) upstream this produces
/// log returns: `Log -> Diff`.
pub struct Diff<T: Scalar + Float>(PhantomData<T>);

impl<T: Scalar + Float> Diff<T> {
    /// Create a new diff operator.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float> Default for Diff<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state: the previous input array, initialised to `NaN` so the
/// first tick's output naturally falls out to `NaN` via NaN propagation.
pub struct DiffState<T: Scalar + Float> {
    prev: Vec<T>,
}

impl<T: Scalar + Float> Operator for Diff<T> {
    type State = DiffState<T>;
    type Inputs = Input<Array<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> (DiffState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let out = Array::from_vec(shape, vec![T::nan(); stride]);
        (
            DiffState {
                prev: vec![T::nan(); stride],
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

        for i in 0..dst.len() {
            dst[i] = src[i] - state.prev[i];
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
    fn diff_basic() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = Diff::<f64>::new().init(&a, Instant::MIN);
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
    fn diff_vector() {
        let mut a = Array::from_vec(&[2], vec![0.0_f64; 2]);
        let (mut state, mut out) = Diff::<f64>::new().init(&a, Instant::MIN);

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
        let (mut state, mut out) = Diff::<f64>::new().init(&a, Instant::MIN);

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
