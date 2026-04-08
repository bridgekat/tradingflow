//! Rolling sum accumulator.
//!
//! O(1) per element per tick via incremental add/subtract with NaN counting.

use num_traits::Float;

use crate::Scalar;

use super::accumulator::Accumulator;

/// Incremental sum accumulator.
///
/// If any value in the window is NaN for a given element position, the
/// output for that position is NaN.
pub struct SumAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    nan_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for SumAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        let stride: usize = input_shape.iter().product();
        Self {
            sum: vec![T::zero(); stride],
            nan_count: vec![0; stride],
        }
    }

    fn add(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if v.is_nan() {
                self.nan_count[j] += 1;
            } else {
                self.sum[j] = self.sum[j] + v;
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        for (j, &v) in element.iter().enumerate() {
            if v.is_nan() {
                self.nan_count[j] -= 1;
            } else {
                self.sum[j] = self.sum[j] - v;
            }
        }
    }

    fn write(&self, _count: usize, output: &mut [T]) {
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nan_count[j] > 0 {
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
    use crate::{Array, Notify, Operator, Series};

    type RollingSum = Rolling<SumAccumulator<f64>>;

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut <RollingSum as Operator>::State,
        out: &mut Array<f64>,
        ts: i64,
        val: f64,
    ) -> bool {
        s.push(ts, &[val]);
        RollingSum::compute(state, (s,), out, ts, &Notify::new(&[], 0))
    }

    #[test]
    fn sum_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::count(3).init((&s,), i64::MIN);

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
        let (mut state, mut out) = RollingSum::count(3).init((&s,), i64::MIN);

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
        let (mut state, mut out) = RollingSum::count(2).init((&s,), i64::MIN);

        s.push(1, &[1.0, 10.0]);
        assert!(!RollingSum::compute(
            &mut state,
            (&s,),
            &mut out,
            1,
            &Notify::new(&[], 0)
        ));
        assert!(out.as_slice()[0].is_nan());
        assert!(out.as_slice()[1].is_nan());

        s.push(2, &[2.0, 20.0]);
        assert!(RollingSum::compute(
            &mut state,
            (&s,),
            &mut out,
            2,
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice(), &[3.0, 30.0]);

        s.push(3, &[3.0, 30.0]);
        assert!(RollingSum::compute(
            &mut state,
            (&s,),
            &mut out,
            3,
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice(), &[5.0, 50.0]);
    }

    #[test]
    fn sum_nan_eviction_restores_correct_sum() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingSum::count(3).init((&s,), i64::MIN);

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
        let (mut state, mut out) = RollingSum::time_delta(200).init((&s,), i64::MIN);

        // ts=100: window [100], sum=1.
        s.push(100, &[1.0]);
        assert!(RollingSum::compute(
            &mut state,
            (&s,),
            &mut out,
            100,
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice()[0], 1.0);

        // ts=200: window [100, 200], sum=3.
        s.push(200, &[2.0]);
        assert!(RollingSum::compute(
            &mut state,
            (&s,),
            &mut out,
            200,
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice()[0], 3.0);

        // ts=350: cutoff=150, evict ts=100. Window [200, 350], sum=5.
        s.push(350, &[3.0]);
        assert!(RollingSum::compute(
            &mut state,
            (&s,),
            &mut out,
            350,
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice()[0], 5.0);
    }
}
