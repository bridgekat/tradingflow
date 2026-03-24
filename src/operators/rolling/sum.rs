//! Rolling sum operator.
//!
//! O(1) per element per tick via incremental add/subtract with NaN counting.

use num_traits::Float;

use crate::{Operator, Scalar, Series};

/// Element-wise rolling sum of last `window` values.
///
/// If any value in the window is NaN, the output for that element is NaN.
pub struct RollingSum<T: Scalar + Float> {
    window: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> RollingSum<T> {
    pub fn new(window: usize) -> Self {
        assert!(window >= 1, "window must be >= 1");
        Self {
            window,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`RollingSum`].
pub struct SumState<T: Scalar + Float> {
    window: usize,
    sum: Vec<T>,
    nan_count: Vec<u32>,
}

impl<T: Scalar + Float> Operator for RollingSum<T> {
    type State = SumState<T>;
    type Inputs = (Series<T>,);
    type Output = Series<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (SumState<T>, Series<T>) {
        let stride = inputs.0.stride();
        let state = SumState {
            window: self.window,
            sum: vec![T::zero(); stride],
            nan_count: vec![0; stride],
        };
        (state, Series::new(inputs.0.shape()))
    }

    fn compute(
        state: &mut SumState<T>,
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
        let mut buf = vec![T::zero(); stride];
        for j in 0..stride {
            buf[j] = if state.nan_count[j] > 0 {
                T::nan()
            } else {
                state.sum[j]
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
        state: &mut SumState<f64>,
        out: &mut Series<f64>,
        ts: i64,
        val: f64,
    ) {
        s.push(ts, &[val]);
        RollingSum::compute(state, (s,), out, ts);
    }

    #[test]
    fn sum_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        assert_eq!(out.last().unwrap()[0], 1.0);

        push_compute(&mut s, &mut state, &mut out, 2, 2.0);
        assert_eq!(out.last().unwrap()[0], 3.0);

        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert_eq!(out.last().unwrap()[0], 6.0);

        push_compute(&mut s, &mut state, &mut out, 4, 4.0);
        assert_eq!(out.last().unwrap()[0], 9.0); // 2+3+4
    }

    #[test]
    fn sum_nan_propagation() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        // Window contains NaN → output NaN
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 4.0);
        // Window [NaN, 3, 4] → still NaN
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 5, 5.0);
        // Window [3, 4, 5] → NaN evicted
        assert_eq!(out.last().unwrap()[0], 12.0);
    }

    #[test]
    fn sum_vector() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingSum::<f64>::new(2).init((&s,), i64::MIN);

        s.push(1, &[1.0, 10.0]);
        RollingSum::compute(&mut state, (&s,), &mut out, 1);
        assert_eq!(out.last().unwrap(), &[1.0, 10.0]);

        s.push(2, &[2.0, 20.0]);
        RollingSum::compute(&mut state, (&s,), &mut out, 2);
        assert_eq!(out.last().unwrap(), &[3.0, 30.0]);

        s.push(3, &[3.0, 30.0]);
        RollingSum::compute(&mut state, (&s,), &mut out, 3);
        assert_eq!(out.last().unwrap(), &[5.0, 50.0]); // 2+3, 20+30
    }

    #[test]
    fn sum_nan_at_start() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::<f64>::new(2).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, f64::NAN);
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 2, 5.0);
        // Window [NaN, 5] → NaN
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 3, 10.0);
        // Window [5, 10] → NaN evicted
        assert_eq!(out.last().unwrap()[0], 15.0);
    }

    #[test]
    fn sum_multiple_nans() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 3, 1.0);
        assert!(out.last().unwrap()[0].is_nan()); // two NaNs in window

        push_compute(&mut s, &mut state, &mut out, 4, 2.0);
        assert!(out.last().unwrap()[0].is_nan()); // still one NaN

        push_compute(&mut s, &mut state, &mut out, 5, 3.0);
        // Window [1, 2, 3] → both NaNs evicted
        assert_eq!(out.last().unwrap()[0], 6.0);
    }

    #[test]
    fn sum_nan_vector_independent() {
        // NaN in element 0 should not affect element 1.
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingSum::<f64>::new(2).init((&s,), i64::MIN);

        s.push(1, &[f64::NAN, 10.0]);
        RollingSum::compute(&mut state, (&s,), &mut out, 1);
        assert!(out.last().unwrap()[0].is_nan());
        assert_eq!(out.last().unwrap()[1], 10.0);

        s.push(2, &[5.0, 20.0]);
        RollingSum::compute(&mut state, (&s,), &mut out, 2);
        assert!(out.last().unwrap()[0].is_nan()); // NaN still in window for elem 0
        assert_eq!(out.last().unwrap()[1], 30.0);

        s.push(3, &[7.0, 30.0]);
        RollingSum::compute(&mut state, (&s,), &mut out, 3);
        assert_eq!(out.last().unwrap()[0], 12.0); // NaN evicted: 5+7
        assert_eq!(out.last().unwrap()[1], 50.0); // 20+30
    }

    #[test]
    fn sum_nan_eviction_restores_correct_sum() {
        // Verify the running sum is accurate after NaN eviction.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 10.0);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 3, 30.0);
        push_compute(&mut s, &mut state, &mut out, 4, 40.0);
        // Window [NaN, 30, 40] → NaN
        assert!(out.last().unwrap()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 5, 50.0);
        // Window [30, 40, 50] → NaN evicted, sum = 120
        assert_eq!(out.last().unwrap()[0], 120.0);
    }
}
