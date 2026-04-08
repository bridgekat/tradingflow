//! Rolling mean accumulator.
//!
//! O(1) per element per tick via incremental add/subtract with NaN counting.

use num_traits::Float;

use crate::Scalar;

use super::accumulator::Accumulator;

/// Incremental mean accumulator.
///
/// If any value in the window is NaN for a given element position, the
/// output for that position is NaN.
pub struct MeanAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    nan_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for MeanAccumulator<T> {
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

    fn write(&self, count: usize, output: &mut [T]) {
        let n = T::from(count).unwrap();
        for (j, o) in output.iter_mut().enumerate() {
            *o = if self.nan_count[j] > 0 {
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
    use crate::{Array, Notify, Operator, Series};

    type RollingMean = Rolling<MeanAccumulator<f64>>;

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut <RollingMean as Operator>::State,
        out: &mut Array<f64>,
        ts: i64,
        val: f64,
    ) -> bool {
        s.push(ts, &[val]);
        RollingMean::compute(state, (s,), out, ts, &Notify::new(&[], 0))
    }

    #[test]
    fn mean_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::count(3).init((&s,), i64::MIN);

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
        let (mut state, mut out) = RollingMean::count(2).init((&s,), i64::MIN);

        assert!(!push_compute(&mut s, &mut state, &mut out, 1, 1.0));
        assert!(push_compute(&mut s, &mut state, &mut out, 2, f64::NAN));
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 4.0);
        assert_eq!(out.as_slice()[0], 3.5);
    }

    #[test]
    fn mean_constant() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingMean::count(5).init((&s,), i64::MIN);

        for i in 1..=10 {
            push_compute(&mut s, &mut state, &mut out, i, 7.0);
        }
        assert!((out.as_slice()[0] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn mean_time_delta() {
        let mut s = Series::<f64>::new(&[]);
        // Window: 200 ns.
        let (mut state, mut out) = RollingMean::time_delta(200).init((&s,), i64::MIN);

        s.push(100, &[2.0]);
        assert!(RollingMean::compute(
            &mut state,
            (&s,),
            &mut out,
            100,
            &Notify::new(&[], 0)
        ));
        assert_eq!(out.as_slice()[0], 2.0); // mean of [2]

        s.push(200, &[4.0]);
        RollingMean::compute(&mut state, (&s,), &mut out, 200, &Notify::new(&[], 0));
        assert_eq!(out.as_slice()[0], 3.0); // mean of [2, 4]

        // ts=350: evict ts=100. Window [200, 350], mean of [4, 6] = 5.
        s.push(350, &[6.0]);
        RollingMean::compute(&mut state, (&s,), &mut out, 350, &Notify::new(&[], 0));
        assert_eq!(out.as_slice()[0], 5.0);
    }
}
