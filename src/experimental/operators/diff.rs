//! Diff operator — element-wise first difference across ticks.  Stateful.

use crate::{Array, Input, Instant};

use crate::experimental::Operator;

/// Element-wise first difference: `input - input_prev`.
///
/// Output is NaN on the first tick (no previous value).
pub struct Diff;

impl Diff {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Diff {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state: the previous input array.
#[derive(Clone)]
pub struct DiffState {
    prev: Vec<f64>,
}

impl Operator for Diff {
    type State = DiffState;
    type Inputs = Input<Array<f64>>;
    type Output = Array<f64>;

    fn is_stateful() -> bool {
        true
    }

    fn init(self, inputs: &Array<f64>, _timestamp: Instant) -> (DiffState, Array<f64>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        (
            DiffState {
                prev: vec![f64::NAN; stride],
            },
            Array::from_vec(shape, vec![f64::NAN; stride]),
        )
    }

    fn compute(
        state: &mut DiffState,
        inputs: &Array<f64>,
        output: &mut Array<f64>,
        _timestamp: Instant,
        _produced: bool,
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

    #[test]
    fn basic() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = Diff::new().init(&a, Instant::MIN);
        assert!(out[0].is_nan());

        a[0] = 10.0;
        Diff::compute(&mut state, &a, &mut out, Instant::from_nanos(1), false);
        assert!(out[0].is_nan()); // first tick: no prev

        a[0] = 25.0;
        Diff::compute(&mut state, &a, &mut out, Instant::from_nanos(2), false);
        assert_eq!(out[0], 15.0);

        a[0] = 20.0;
        Diff::compute(&mut state, &a, &mut out, Instant::from_nanos(3), false);
        assert_eq!(out[0], -5.0);
    }
}
