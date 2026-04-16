//! Rolling sum accumulator.
//!
//! O(1) per element per tick via incremental add/subtract with non-finite
//! counting.

use num_traits::Float;


use crate::Scalar;

use super::accumulator::Accumulator;

/// Incremental sum accumulator.
///
/// Non-finite values (NaN, ±inf) are skipped and counted separately rather
/// than added to the running sum, since `inf − inf` would corrupt the sum
/// to NaN on eviction.  If any value in the window is non-finite for a
/// given element position, the output for that position is NaN.
pub struct SumAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for SumAccumulator<T> {
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

    fn write(&self, _count: usize, output: &mut [T]) {
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] > 0 {
                T::nan()
            } else {
                self.sum[j]
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::rolling::accumulator::Rolling;
    use crate::{Duration, Instant};
    use crate::{Array, Notify, Operator, Series};

    type RollingSum = Rolling<SumAccumulator<f64>>;

    fn ts(n: i64) -> Instant { Instant::from_nanos(n) }

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut <RollingSum as Operator>::State,
        out: &mut Array<f64>,
        t: i64,
        val: f64,
    ) -> bool {
        s.push(ts(t), &[val]);
        RollingSum::compute(state, s, out, ts(t), &Notify::new(&[], 0))
    }

    #[test]
    fn sum_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::count(3).init(&s, Instant::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 1.0));
        assert!(out.as_slice()[0].is_nan());

        assert!(!push_compute(&mut s, &mut state, &mut out, 2, 2.0));
        assert!(out.as_slice()[0].is_nan());

        assert!(push_compute(&mut s, &mut state, &mut out, 3, 3.0));
        assert_eq!(out.as_slice()[0], 6.0);

        assert!(push_compute(&mut s, &mut state, &mut out, 4, 4.0));
        assert_eq!(out.as_slice()[0], 9.0);
    }

    #[test]
    fn sum_nan_propagation() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::count(3).init(&s, Instant::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 1.0));
        assert!(!push_compute(&mut s, &mut state, &mut out, 2, f64::NAN));
        assert!(push_compute(&mut s, &mut state, &mut out, 3, 3.0));
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 4.0);
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 5, 5.0);
        assert_eq!(out.as_slice()[0], 12.0);
    }

    #[test]
    fn sum_vector() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingSum::count(2).init(&s, Instant::MIN);

        s.push(ts(1), &[1.0, 10.0]);
        assert!(!RollingSum::compute(
            &mut state,
            &s,
            &mut out,
            ts(1),
            &Notify::new(&[], 0)
        ));
        assert!(out.as_slice()[0].is_nan());
        assert!(out.as_slice()[1].is_nan());

        s.push(ts(2), &[2.0, 20.0]);
        assert!(RollingSum::compute(
            &mut state,
            &s,
            &mut out,
            ts(2),
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice(), &[3.0, 30.0]);

        s.push(ts(3), &[3.0, 30.0]);
        assert!(RollingSum::compute(
            &mut state,
            &s,
            &mut out,
            ts(3),
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice(), &[5.0, 50.0]);
    }

    #[test]
    fn sum_nan_eviction_restores_correct_sum() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::count(3).init(&s, Instant::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 10.0));
        assert!(!push_compute(&mut s, &mut state, &mut out, 2, f64::NAN));
        assert!(push_compute(&mut s, &mut state, &mut out, 3, 30.0));
        assert!(out.as_slice()[0].is_nan());
        push_compute(&mut s, &mut state, &mut out, 4, 40.0);
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 5, 50.0);
        assert_eq!(out.as_slice()[0], 120.0);
    }

    #[test]
    fn sum_time_delta() {
        let mut s = Series::<f64>::new(&[]);
        // Window: 200 ns.
        let (mut state, mut out) = RollingSum::time_delta(Duration::from_nanos(200)).init(&s, Instant::MIN);

        // ts=100: window [100], sum=1.
        s.push(ts(100), &[1.0]);
        assert!(RollingSum::compute(
            &mut state,
            &s,
            &mut out,
            ts(100),
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice()[0], 1.0);

        // ts=200: window [100, 200], sum=3.
        s.push(ts(200), &[2.0]);
        assert!(RollingSum::compute(
            &mut state,
            &s,
            &mut out,
            ts(200),
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice()[0], 3.0);

        // ts=350: cutoff=150, evict ts=100. Window [200, 350], sum=5.
        s.push(ts(350), &[3.0]);
        assert!(RollingSum::compute(
            &mut state,
            &s,
            &mut out,
            ts(350),
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice()[0], 5.0);
    }
}
