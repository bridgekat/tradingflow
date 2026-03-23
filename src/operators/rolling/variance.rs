//! Rolling variance operator.

use num_traits::Float;

use crate::operator::Operator;
use crate::series::Series;
use crate::types::Scalar;

/// Element-wise rolling variance of last `window` values.
///
/// Uses the formula `Var(x) = E[x^2] - E[x]^2` (population variance).
/// If any value in the window is NaN, the output for that element is NaN.
pub struct RollingVariance<T: Scalar + Float> {
    window: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> RollingVariance<T> {
    pub fn new(window: usize) -> Self {
        assert!(window >= 1, "window must be >= 1");
        Self {
            window,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float> Operator for RollingVariance<T> {
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

        let mut sum = vec![T::zero(); stride];
        let mut sum_sq = vec![T::zero(); stride];

        for i in start..len {
            let row = series.at(i);
            for (j, &v) in row.iter().enumerate() {
                if v.is_nan() {
                    sum[j] = T::nan();
                    sum_sq[j] = T::nan();
                } else if !sum[j].is_nan() {
                    sum[j] = sum[j] + v;
                    sum_sq[j] = sum_sq[j] + v * v;
                }
            }
        }

        let mut buf = vec![T::nan(); stride];
        for j in 0..stride {
            if !sum[j].is_nan() {
                let mean = sum[j] / count;
                buf[j] = sum_sq[j] / count - mean * mean;
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
        RollingVariance::compute(state, (s,), out, ts);
    }

    #[test]
    fn variance_constant() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(5).init((&s,), i64::MIN);

        for i in 1..=5 {
            push_compute(&mut s, &mut state, &mut out, i, 10.0);
        }
        assert!((out.last().unwrap()[0]).abs() < 1e-10); // variance of constant = 0
    }

    #[test]
    fn variance_known() {
        // Var([1,2,3]) = E[x^2] - E[x]^2 = (1+4+9)/3 - (6/3)^2 = 14/3 - 4 = 2/3
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        push_compute(&mut s, &mut state, &mut out, 2, 2.0);
        push_compute(&mut s, &mut state, &mut out, 3, 3.0);

        let var = out.last().unwrap()[0];
        assert!((var - 2.0 / 3.0).abs() < 1e-10, "expected 2/3, got {var}");
    }

    #[test]
    fn variance_sliding() {
        // After window slides: Var([2,3,4]) = Var([1,2,3]) = 2/3
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(3).init((&s,), i64::MIN);

        for i in 1..=4 {
            push_compute(&mut s, &mut state, &mut out, i, i as f64);
        }
        let var = out.last().unwrap()[0];
        assert!((var - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn variance_nan() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert!(out.last().unwrap()[0].is_nan());
    }

    #[test]
    fn variance_vector() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingVariance::<f64>::new(2).init((&s,), i64::MIN);

        s.push(1, &[1.0, 10.0]);
        RollingVariance::compute(&mut state, (&s,), &mut out, 1);
        s.push(2, &[3.0, 20.0]);
        RollingVariance::compute(&mut state, (&s,), &mut out, 2);

        let row = out.last().unwrap();
        // Var([1,3]) = (1+9)/2 - (4/2)^2 = 5 - 4 = 1
        assert!((row[0] - 1.0).abs() < 1e-10);
        // Var([10,20]) = (100+400)/2 - (30/2)^2 = 250 - 225 = 25
        assert!((row[1] - 25.0).abs() < 1e-10);
    }
}
