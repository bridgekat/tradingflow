//! Rolling variance accumulator.
//!
//! O(1) per element per tick via incremental sum/sum_sq with non-finite
//! counting.

use num_traits::Float;


use crate::Scalar;

use super::accumulator::Accumulator;

/// Incremental population variance accumulator.
///
/// Uses the formula `Var(x) = E[x²] − E[x]²`.
/// Non-finite values (NaN, ±inf) are skipped and counted separately rather
/// than added to the running sums, since `inf − inf` would corrupt the
/// sums to NaN on eviction.  If any value in the window is non-finite for
/// a given element position, the output for that position is NaN.
pub struct VarianceAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    sum_sq: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for VarianceAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        let stride: usize = input_shape.iter().product();
        Self {
            sum: vec![T::zero(); stride],
            sum_sq: vec![T::zero(); stride],
            nonfinite_count: vec![0; stride],
        }
    }

    fn add(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] += 1;
            } else {
                self.sum[j] = self.sum[j] + v;
                self.sum_sq[j] = self.sum_sq[j] + v * v;
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if !v.is_finite() {
                self.nonfinite_count[j] -= 1;
            } else {
                self.sum[j] = self.sum[j] - v;
                self.sum_sq[j] = self.sum_sq[j] - v * v;
            }
        }
    }

    fn write(&self, count: usize, output: &mut [T]) {
        let n = T::from(count).unwrap();
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nonfinite_count[j] == 0 {
                let mean = self.sum[j] / n;
                self.sum_sq[j] / n - mean * mean
            } else {
                T::nan()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::rolling::accumulator::Rolling;
    use crate::time::{Duration, Instant};
    use crate::{Array, Notify, Operator, Series};

    type RollingVariance = Rolling<VarianceAccumulator<f64>>;

    fn ts(n: i64) -> Instant { Instant::from_nanos(n) }

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut <RollingVariance as Operator>::State,
        out: &mut Array<f64>,
        t: i64,
        val: f64,
    ) -> bool {
        s.push(ts(t), &[val]);
        RollingVariance::compute(state, (s,), out, ts(t), &Notify::new(&[], 0))
    }

    #[test]
    fn var_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::count(3).init((&s,), Instant::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 1.0));
        assert!(!push_compute(&mut s, &mut state, &mut out, 2, 2.0));

        assert!(push_compute(&mut s, &mut state, &mut out, 3, 3.0));
        // Var([1,2,3]) = E[x²] - E[x]² = (1+4+9)/3 - 4 = 14/3 - 4 = 2/3
        assert!((out.as_slice()[0] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn var_nan() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::count(2).init((&s,), Instant::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 1.0));
        assert!(push_compute(&mut s, &mut state, &mut out, 2, f64::NAN));
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 5.0);
        // Var([3,5]) = (9+25)/2 - 16 = 1.0
        assert!((out.as_slice()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn var_time_delta() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::time_delta(Duration::from_nanos(200)).init((&s,), Instant::MIN);

        s.push(ts(100), &[2.0]);
        assert!(RollingVariance::compute(
            &mut state,
            (&s,),
            &mut out,
            ts(100),
            &Notify::new(&[], 0)
        ));
        // Single element → variance = 0.
        assert_eq!(out.as_slice()[0], 0.0);

        s.push(ts(200), &[4.0]);
        RollingVariance::compute(&mut state, (&s,), &mut out, ts(200), &Notify::new(&[], 0));
        // Var([2,4]) = (4+16)/2 - 9 = 1.0
        assert!((out.as_slice()[0] - 1.0).abs() < 1e-10);
    }
}
