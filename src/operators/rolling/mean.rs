//! Rolling mean accumulator.
//!
//! O(1) per element per tick via incremental add/subtract with non-finite
//! counting.

use num_traits::Float;


use crate::Scalar;

use super::accumulator::Accumulator;

/// Incremental mean accumulator.
///
/// Non-finite values (NaN, ±inf) are skipped and counted separately rather
/// than added to the running sum, since `inf − inf` would corrupt the sum
/// to NaN on eviction.  If any value in the window is non-finite for a
/// given element position, the output for that position is NaN.
pub struct MeanAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for MeanAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        let stride: usize = input_shape.iter().product();
        Self {
            sum: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
        }
    }

    fn add(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] += 1;
            } else {
                self.sum[j] = self.sum[j] + v;
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] -= 1;
            } else {
                self.sum[j] = self.sum[j] - v;
            }
        }
    }

    fn write(&self, count: usize, output: &mut [T]) {
        let n = T::from(count).unwrap();
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] > 0 {
                T::nan()
            } else {
                self.sum[j] / n
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::rolling::accumulator::Rolling;
    use crate::{Duration, Instant};
    use crate::{Array, Operator, Series};

    type RollingMean = Rolling<MeanAccumulator<f64>>;

    fn ts(n: i64) -> Instant { Instant::from_nanos(n) }

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut <RollingMean as Operator>::State,
        out: &mut Array<f64>,
        t: i64,
        val: f64,
    ) -> bool {
        s.push(ts(t), &[val]);
        RollingMean::compute(state, s, out, ts(t), false)
    }

    #[test]
    fn mean_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::count(3).init(&s, Instant::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 1.0));
        assert!(!push_compute(&mut s, &mut state, &mut out, 2, 2.0));

        assert!(push_compute(&mut s, &mut state, &mut out, 3, 3.0));
        assert_eq!(out.as_slice()[0], 2.0);

        assert!(push_compute(&mut s, &mut state, &mut out, 4, 6.0));
        assert_eq!(out.as_slice()[0], 11.0 / 3.0);
    }

    #[test]
    fn mean_nan() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::count(2).init(&s, Instant::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 1.0));
        assert!(push_compute(&mut s, &mut state, &mut out, 2, f64::NAN));
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 4.0);
        assert_eq!(out.as_slice()[0], 3.5);
    }

    #[test]
    fn mean_inf() {
        // Regression: inf should be treated like NaN, not added to sum,
        // because `inf - inf` on eviction would corrupt the running sum
        // to NaN forever.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::count(2).init(&s, Instant::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, f64::INFINITY);
        push_compute(&mut s, &mut state, &mut out, 2, 2.0);
        // Window [inf, 2.0] → mean is NaN (one non-finite present).
        assert!(out.as_slice()[0].is_nan());

        // Window [2.0, 3.0] → inf evicted, mean should be 2.5.
        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert_eq!(out.as_slice()[0], 2.5);

        // Same for -inf.
        push_compute(&mut s, &mut state, &mut out, 4, f64::NEG_INFINITY);
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 5, 5.0);
        push_compute(&mut s, &mut state, &mut out, 6, 6.0);
        assert_eq!(out.as_slice()[0], 5.5);
    }

    #[test]
    fn mean_constant() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::count(5).init(&s, Instant::MIN);

        for i in 1..=10 {
            push_compute(&mut s, &mut state, &mut out, i, 7.0);
        }
        assert!((out.as_slice()[0] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn mean_time_delta() {
        let mut s = Series::<f64>::new(&[]);
        // Window: 200 ns.
        let (mut state, mut out) = RollingMean::time_delta(Duration::from_nanos(200)).init(&s, Instant::MIN);

        s.push(ts(100), &[2.0]);
        assert!(RollingMean::compute(
            &mut state,
            &s,
            &mut out,
            ts(100),
            false
        ));
        assert_eq!(out.as_slice()[0], 2.0); // mean of [2]

        s.push(ts(200), &[4.0]);
        RollingMean::compute(&mut state, &s, &mut out, ts(200), false);
        assert_eq!(out.as_slice()[0], 3.0); // mean of [2, 4]

        // ts=350: evict ts=100. Window [200, 350], mean of [4, 6] = 5.
        s.push(ts(350), &[6.0]);
        RollingMean::compute(&mut state, &s, &mut out, ts(350), false);
        assert_eq!(out.as_slice()[0], 5.0);
    }
}
