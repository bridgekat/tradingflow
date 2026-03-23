//! Rolling mean operator.

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
    pub fn new(window: usize) -> Self {
        assert!(window >= 1, "window must be >= 1");
        Self {
            window,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float> Operator for RollingMean<T> {
    type State = usize;
    type Inputs = (Series<T>,);
    type Output = Series<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (usize, Series<T>) {
        (self.window, Series::new(inputs.0.shape()))
    }

    fn compute(
        state: &mut usize,
        inputs: (&Series<T>,),
        output: &mut Series<T>,
        timestamp: i64,
    ) -> bool {
        let window = *state;
        let series = inputs.0;
        let len = series.len();
        let stride = series.stride();
        let start = len.saturating_sub(window);
        let count = T::from(len - start).unwrap();

        let mut buf = vec![T::zero(); stride];

        for i in start..len {
            let row = series.at(i);
            for (j, &v) in row.iter().enumerate() {
                if v.is_nan() {
                    buf[j] = T::nan();
                } else if !buf[j].is_nan() {
                    buf[j] = buf[j] + v;
                }
            }
        }

        for v in buf.iter_mut() {
            if !v.is_nan() {
                *v = *v / count;
            }
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
        state: &mut usize,
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
}
