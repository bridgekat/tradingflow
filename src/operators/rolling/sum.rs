//! Rolling sum operator.

use num_traits::Float;

use crate::operator::Operator;
use crate::series::Series;
use crate::types::Scalar;

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

impl<T: Scalar + Float> Operator for RollingSum<T> {
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
}
