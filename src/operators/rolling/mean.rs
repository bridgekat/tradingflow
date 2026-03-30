//! Rolling mean operator.
//!
//! O(1) per element per tick via incremental add/subtract with NaN counting.

use num_traits::Float;

use crate::{Operator, Scalar, Series};

/// Element-wise rolling mean of last `window` values.
///
/// If any value in the window is NaN, the output for that element is NaN.
pub struct RollingMean<T: Scalar + Float> {
    window: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> RollingMean<T> {
    /// Create a new rolling mean operator with the given window size.
    pub fn new(window: usize) -> Self {
        assert!(window >= 1, "window must be >= 1");
        Self {
            window,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`RollingMean`].
pub struct MeanState<T: Scalar + Float> {
    window: usize,
    /// Running sum per element position.
    sum: Vec<T>,
    /// NaN count in window per element position.
    nan_count: Vec<u32>,
}

impl<T: Scalar + Float> Operator for RollingMean<T> {
    type State = MeanState<T>;
    type Inputs = (Series<T>,);
    type Output = Series<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (MeanState<T>, Series<T>) {
        let stride = inputs.0.stride();
        let state = MeanState {
            window: self.window,
            sum: vec![T::zero(); stride],
            nan_count: vec![0; stride],
        };
        (state, Series::new(inputs.0.shape()))
    }

    fn compute(
        state: &mut MeanState<T>,
        inputs: (&Series<T>,),
        output: &mut Series<T>,
        timestamp: i64,
    ) -> bool {
        let series = inputs.0;
        let len = series.len();
        let stride = state.sum.len();

        // Add new element.
        let new_row = series.at(len - 1);
        for j in 0..stride {
            let v = new_row[j];
            if v.is_nan() {
                state.nan_count[j] += 1;
            } else {
                state.sum[j] = state.sum[j] + v;
            }
        }

        // Evict oldest element if window is full.
        if len > state.window {
            let old_row = series.at(len - 1 - state.window);
            for j in 0..stride {
                let v = old_row[j];
                if v.is_nan() {
                    state.nan_count[j] -= 1;
                } else {
                    state.sum[j] = state.sum[j] - v;
                }
            }
        }

        // Produce output.
        let count = T::from(len.min(state.window)).unwrap();
        let mut buf = vec![T::zero(); stride];
        for j in 0..stride {
            buf[j] = if state.nan_count[j] > 0 {
                T::nan()
            } else {
                state.sum[j] / count
            };
        }

        output.push(timestamp, &buf);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut MeanState<f64>,
        out: &mut Series<f64>,
        ts: i64,
        val: f64,
    ) {
        s.push(ts, &[val]);
        RollingMean::compute(state, (s,), out, ts);
    }

    #[test]
    fn mean_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        assert_eq!(out.last().unwrap()[0], 1.0); // mean of [1]

        push_compute(&mut s, &mut state, &mut out, 2, 2.0);
        assert_eq!(out.last().unwrap()[0], 1.5); // mean of [1,2]

        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert_eq!(out.last().unwrap()[0], 2.0); // mean of [1,2,3]

        push_compute(&mut s, &mut state, &mut out, 4, 6.0);
        assert_eq!(out.last().unwrap()[0], 11.0 / 3.0); // mean of [2,3,6]
    }

    #[test]
    fn mean_nan() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::<f64>::new(2).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        // Window [NaN, 3] → NaN
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 4.0);
        // Window [3, 4] → 3.5
        assert_eq!(out.last().unwrap()[0], 3.5);
    }

    #[test]
    fn mean_constant() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::<f64>::new(5).init((&s,), i64::MIN);

        for i in 1..=10 {
            push_compute(&mut s, &mut state, &mut out, i, 7.0);
        }
        assert!((out.last().unwrap()[0] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn mean_nan_eviction_restores() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 6.0);
        assert!(out.last().unwrap()[0].is_nan()); // NaN still in window

        push_compute(&mut s, &mut state, &mut out, 5, 9.0);
        // Window [3, 6, 9] → NaN evicted, mean = 6
        assert_eq!(out.last().unwrap()[0], 6.0);
    }

    #[test]
    fn mean_multiple_nans_in_window() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 3, 6.0);
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 9.0);
        assert!(out.last().unwrap()[0].is_nan()); // one NaN remains

        push_compute(&mut s, &mut state, &mut out, 5, 12.0);
        // Window [6, 9, 12] → both NaNs evicted, mean = 9
        assert_eq!(out.last().unwrap()[0], 9.0);
    }

    #[test]
    fn mean_nan_vector_independent() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingMean::<f64>::new(2).init((&s,), i64::MIN);

        s.push(1, &[f64::NAN, 4.0]);
        RollingMean::compute(&mut state, (&s,), &mut out, 1);
        assert!(out.last().unwrap()[0].is_nan());
        assert_eq!(out.last().unwrap()[1], 4.0);

        s.push(2, &[6.0, 8.0]);
        RollingMean::compute(&mut state, (&s,), &mut out, 2);
        assert!(out.last().unwrap()[0].is_nan());
        assert_eq!(out.last().unwrap()[1], 6.0); // (4+8)/2

        s.push(3, &[10.0, 12.0]);
        RollingMean::compute(&mut state, (&s,), &mut out, 3);
        assert_eq!(out.last().unwrap()[0], 8.0); // (6+10)/2, NaN evicted
        assert_eq!(out.last().unwrap()[1], 10.0); // (8+12)/2
    }
}
