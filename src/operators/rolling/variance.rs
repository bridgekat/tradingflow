//! Rolling variance operator.
//!
//! O(1) per element per tick via incremental sum/sum_sq with NaN counting.

use num_traits::Float;

use crate::{Array, Notify, Operator, Scalar, Series};

/// Element-wise rolling variance of last `window` values.
///
/// Uses the formula `Var(x) = E[x^2] - E[x]^2` (population variance).
/// If any value in the window is NaN, the output for that element is NaN.
pub struct RollingVariance<T: Scalar + Float> {
    window: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> RollingVariance<T> {
    /// Create a new rolling variance operator with the given window size.
    pub fn new(window: usize) -> Self {
        assert!(window >= 1, "window must be >= 1");
        Self {
            window,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`RollingVariance`].
pub struct VarState<T: Scalar + Float> {
    window: usize,
    /// Running sum per element position.
    sum: Vec<T>,
    /// Running sum of squares per element position.
    sum_sq: Vec<T>,
    /// NaN count in window per element position.
    nan_count: Vec<u32>,
}

impl<T: Scalar + Float> Operator for RollingVariance<T> {
    type State = VarState<T>;
    type Inputs = (Series<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (VarState<T>, Array<T>) {
        let stride = inputs.0.stride();
        let state = VarState {
            window: self.window,
            sum: vec![T::zero(); stride],
            sum_sq: vec![T::zero(); stride],
            nan_count: vec![0; stride],
        };
        (state, Array::zeros(inputs.0.shape()))
    }

    fn compute(
        state: &mut VarState<T>,
        inputs: (&Series<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
        _notify: &Notify<'_>,
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
                state.sum_sq[j] = state.sum_sq[j] + v * v;
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
                    state.sum_sq[j] = state.sum_sq[j] - v * v;
                }
            }
        }

        // Produce output.
        let count = T::from(len.min(state.window)).unwrap();
        let out = output.as_mut_slice();
        for j in 0..stride {
            out[j] = if state.nan_count[j] == 0 {
                let mean = state.sum[j] / count;
                state.sum_sq[j] / count - mean * mean
            } else {
                T::nan()
            };
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_compute(
        s: &mut Series<f64>,
        state: &mut VarState<f64>,
        out: &mut Array<f64>,
        ts: i64,
        val: f64,
    ) {
        s.push(ts, &[val]);
        RollingVariance::compute(state, (s,), out, ts, &Notify::new(&[], &[]));
    }

    #[test]
    fn variance_constant() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(5).init((&s,), i64::MIN);

        for i in 1..=5 {
            push_compute(&mut s, &mut state, &mut out, i, 10.0);
        }
        assert!((out.as_slice()[0]).abs() < 1e-10); // variance of constant = 0
    }

    #[test]
    fn variance_known() {
        // Var([1,2,3]) = E[x^2] - E[x]^2 = (1+4+9)/3 - (6/3)^2 = 14/3 - 4 = 2/3
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        push_compute(&mut s, &mut state, &mut out, 2, 2.0);
        push_compute(&mut s, &mut state, &mut out, 3, 3.0);

        let var = out.as_slice()[0];
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
        let var = out.as_slice()[0];
        assert!((var - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn variance_nan() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert!(out.as_slice()[0].is_nan());
    }

    #[test]
    fn variance_vector() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingVariance::<f64>::new(2).init((&s,), i64::MIN);

        s.push(1, &[1.0, 10.0]);
        RollingVariance::compute(&mut state, (&s,), &mut out, 1, &Notify::new(&[], &[]));
        s.push(2, &[3.0, 20.0]);
        RollingVariance::compute(&mut state, (&s,), &mut out, 2, &Notify::new(&[], &[]));

        let row = out.as_slice();
        // Var([1,3]) = (1+9)/2 - (4/2)^2 = 5 - 4 = 1
        assert!((row[0] - 1.0).abs() < 1e-10);
        // Var([10,20]) = (100+400)/2 - (30/2)^2 = 250 - 225 = 25
        assert!((row[1] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn variance_nan_eviction() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(3).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, 1.0);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 4, 4.0);
        assert!(out.as_slice()[0].is_nan()); // NaN still in window

        push_compute(&mut s, &mut state, &mut out, 5, 5.0);
        // Window [3, 4, 5] → Var = E[x²]-E[x]² = (9+16+25)/3 - (12/3)² = 50/3 - 16 = 2/3
        let var = out.as_slice()[0];
        assert!(
            (var - 2.0 / 3.0).abs() < 1e-10,
            "expected 2/3 after NaN eviction, got {var}"
        );
    }

    #[test]
    fn variance_multiple_nans() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = RollingVariance::<f64>::new(2).init((&s,), i64::MIN);

        push_compute(&mut s, &mut state, &mut out, 1, f64::NAN);
        push_compute(&mut s, &mut state, &mut out, 2, f64::NAN);
        assert!(out.as_slice()[0].is_nan());

        push_compute(&mut s, &mut state, &mut out, 3, 3.0);
        assert!(out.as_slice()[0].is_nan()); // one NaN remains

        push_compute(&mut s, &mut state, &mut out, 4, 5.0);
        // Window [3, 5] → Var = (9+25)/2 - (8/2)² = 17 - 16 = 1
        let var = out.as_slice()[0];
        assert!((var - 1.0).abs() < 1e-10);
    }

    #[test]
    fn variance_nan_vector_independent() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingVariance::<f64>::new(2).init((&s,), i64::MIN);

        s.push(1, &[f64::NAN, 2.0]);
        RollingVariance::compute(&mut state, (&s,), &mut out, 1, &Notify::new(&[], &[]));
        assert!(out.as_slice()[0].is_nan());
        assert_eq!(out.as_slice()[1], 0.0); // single value → var = 0

        s.push(2, &[4.0, 4.0]);
        RollingVariance::compute(&mut state, (&s,), &mut out, 2, &Notify::new(&[], &[]));
        assert!(out.as_slice()[0].is_nan()); // NaN still in window
        let var_1 = out.as_slice()[1];
        // Var([2,4]) = (4+16)/2 - (6/2)² = 10 - 9 = 1
        assert!((var_1 - 1.0).abs() < 1e-10);

        s.push(3, &[6.0, 6.0]);
        RollingVariance::compute(&mut state, (&s,), &mut out, 3, &Notify::new(&[], &[]));
        let var_0 = out.as_slice()[0];
        // Window for elem 0: [4, 6] → Var = (16+36)/2 - (10/2)² = 26 - 25 = 1
        assert!((var_0 - 1.0).abs() < 1e-10, "expected 1.0, got {var_0}");
    }
}
