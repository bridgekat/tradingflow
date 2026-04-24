//! Rolling operators — simplified port of `src/operators/rolling/`.
//!
//! This PoC includes only [`RollingMean`] with a count-based window, via
//! an inlined mean accumulator (no separate trait abstraction).  The
//! goal here is to exercise the `Series<T>` → `Array<T>` data-flow shape
//! and the stateful self-edge in the scheduler.
//!
//! Semantics match the legacy `RollingMean::count(w)`: output is NaN
//! until the window is full; non-finite values (NaN, ±inf) are counted
//! separately and poison the output for that element position.

use num_traits::Float;

use super::super::data::{Array, Input, InputTypes, Instant, Scalar, Series};
use super::super::operator::Operator;

/// Count-window rolling mean.
pub struct RollingMean<T: Scalar + Float> {
    window: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> RollingMean<T> {
    /// Create a count-based rolling mean with window `w`.
    pub fn count(window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        Self {
            window,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct RollingMeanState<T: Scalar + Float> {
    window: usize,
    /// Index of the oldest element currently in the window.
    start: usize,
    /// Number of elements currently in the window.
    count: usize,
    /// Running sum per element position (`stride` entries).
    sum: Vec<T>,
    /// Count of non-finite values per element position.
    nonfinite: Vec<u32>,
    stride: usize,
}

impl<T: Scalar + Float> Operator for RollingMean<T> {
    type State = RollingMeanState<T>;
    type Inputs = Input<Series<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Series<T>, _timestamp: Instant) -> (RollingMeanState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let state = RollingMeanState {
            window: self.window,
            start: 0,
            count: 0,
            sum: vec![T::zero(); stride],
            nonfinite: vec![0; stride],
            stride,
        };
        let out = Array::from_vec(shape, vec![T::nan(); stride]);
        (state, out)
    }

    fn compute(
        state: &mut RollingMeanState<T>,
        inputs: &Series<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        let series = inputs;
        let len = series.len();
        if len == 0 {
            return false;
        }
        // Incorporate the newest element at index len-1.
        let new_el = series.at(len - 1);
        for (j, v) in new_el.iter().enumerate() {
            if !v.is_finite() {
                state.nonfinite[j] += 1;
            } else {
                state.sum[j] = state.sum[j] + *v;
            }
        }
        state.count += 1;

        // Evict oldest while window exceeds target size.
        while state.count > state.window {
            let old = series.at(state.start);
            for (j, v) in old.iter().enumerate() {
                if !v.is_finite() {
                    state.nonfinite[j] -= 1;
                } else {
                    state.sum[j] = state.sum[j] - *v;
                }
            }
            state.start += 1;
            state.count -= 1;
        }

        if state.count < state.window {
            // Not enough elements yet.
            return false;
        }

        let n = T::from(state.count).unwrap();
        let out = output.as_mut_slice();
        for j in 0..state.stride {
            out[j] = if state.nonfinite[j] > 0 {
                T::nan()
            } else {
                state.sum[j] / n
            };
        }
        true
    }
}
