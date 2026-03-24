//! Rolling covariance matrix operator.
//!
//! O(K²) per tick via incremental cross-product sums with NaN counting.

use num_traits::Float;

use crate::{Operator, Scalar, Series};

/// Pairwise rolling covariance matrix of last `window` values.
///
/// Input must be 1D with shape `[K]`. Output has shape `[K, K]`.
///
/// `Cov(i,j) = E[x_i * x_j] - E[x_i] * E[x_j]` over the window.
/// If any value in the window is NaN for either element i or j,
/// the output `Cov(i,j)` is NaN.
pub struct RollingCovariance<T: Scalar + Float> {
    window: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> RollingCovariance<T> {
    pub fn new(window: usize) -> Self {
        assert!(window >= 1, "window must be >= 1");
        Self {
            window,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`RollingCovariance`].
pub struct CovState<T: Scalar + Float> {
    window: usize,
    k: usize,
    /// Running sum per element, shape `[K]`.
    sum: Vec<T>,
    /// Running cross-product sums, shape `[K, K]` (flat).
    sum_cross: Vec<T>,
    /// NaN count in window per element, shape `[K]`.
    nan_count: Vec<u32>,
}

impl<T: Scalar + Float> Operator for RollingCovariance<T> {
    type State = CovState<T>;
    type Inputs = (Series<T>,);
    type Output = Series<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (CovState<T>, Series<T>) {
        let shape = inputs.0.shape();
        assert_eq!(
            shape.len(),
            1,
            "RollingCovariance requires 1D input, got shape {:?}",
            shape,
        );
        let k = shape[0];
        let state = CovState {
            window: self.window,
            k,
            sum: vec![T::zero(); k],
            sum_cross: vec![T::zero(); k * k],
            nan_count: vec![0; k],
        };
        (state, Series::new(&[k, k]))
    }

    fn compute(
        state: &mut CovState<T>,
        inputs: (&Series<T>,),
        output: &mut Series<T>,
        timestamp: i64,
    ) -> bool {
        let series = inputs.0;
        let len = series.len();
        let k = state.k;

        // Add new row.
        let new_row = series.at(len - 1);
        for i in 0..k {
            let xi = new_row[i];
            if xi.is_nan() {
                state.nan_count[i] += 1;
            } else {
                state.sum[i] = state.sum[i] + xi;
            }
        }
        for i in 0..k {
            let xi = new_row[i];
            if xi.is_nan() {
                continue;
            }
            for j in i..k {
                let xj = new_row[j];
                if xj.is_nan() {
                    continue;
                }
                let prod = xi * xj;
                state.sum_cross[i * k + j] = state.sum_cross[i * k + j] + prod;
                if i != j {
                    state.sum_cross[j * k + i] = state.sum_cross[j * k + i] + prod;
                }
            }
        }

        // Evict oldest row if window is full.
        if len > state.window {
            let old_row = series.at(len - 1 - state.window);
            for i in 0..k {
                let xi = old_row[i];
                if xi.is_nan() {
                    state.nan_count[i] -= 1;
                } else {
                    state.sum[i] = state.sum[i] - xi;
                }
            }
            for i in 0..k {
                let xi = old_row[i];
                if xi.is_nan() {
                    continue;
                }
                for j in i..k {
                    let xj = old_row[j];
                    if xj.is_nan() {
                        continue;
                    }
                    let prod = xi * xj;
                    state.sum_cross[i * k + j] = state.sum_cross[i * k + j] - prod;
                    if i != j {
                        state.sum_cross[j * k + i] = state.sum_cross[j * k + i] - prod;
                    }
                }
            }
        }

        // Produce output.
        let count = T::from(len.min(state.window)).unwrap();
        let mut buf = vec![T::nan(); k * k];
        for i in 0..k {
            for j in 0..k {
                if state.nan_count[i] == 0 && state.nan_count[j] == 0 {
                    let mean_i = state.sum[i] / count;
                    let mean_j = state.sum[j] / count;
                    buf[i * k + j] = state.sum_cross[i * k + j] / count - mean_i * mean_j;
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

    #[test]
    fn cov_identity() {
        // Two perfectly correlated variables: x = [1,2,3], y = [2,4,6]
        // Cov(x,y) = E[xy] - E[x]E[y]
        //   = (2+8+18)/3 - (6/3)(12/3) = 28/3 - 8 = 4/3
        // Var(x) = 2/3, Var(y) = 8/3
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::<f64>::new(3).init((&s,), i64::MIN);

        s.push(1, &[1.0, 2.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 1);
        s.push(2, &[2.0, 4.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 2);
        s.push(3, &[3.0, 6.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 3);

        let cov = out.last().unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        // Var(x) = 2/3
        assert!((cov[0] - 2.0 / 3.0).abs() < 1e-10, "Var(x) = {}", cov[0]);
        // Cov(x,y) = 4/3
        assert!((cov[1] - 4.0 / 3.0).abs() < 1e-10, "Cov(x,y) = {}", cov[1]);
        // Symmetric
        assert!(
            (cov[2] - cov[1]).abs() < 1e-10,
            "Cov(y,x) should equal Cov(x,y)"
        );
        // Var(y) = 8/3
        assert!((cov[3] - 8.0 / 3.0).abs() < 1e-10, "Var(y) = {}", cov[3]);
    }

    #[test]
    fn cov_output_shape() {
        let mut s = Series::<f64>::new(&[4]);
        let (mut state, mut out) = RollingCovariance::<f64>::new(5).init((&s,), i64::MIN);

        s.push(1, &[1.0, 2.0, 3.0, 4.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 1);

        assert_eq!(out.shape(), &[4, 4]);
        assert_eq!(out.stride(), 16);
    }

    #[test]
    fn cov_uncorrelated() {
        // x = [1, -1, 1, -1], y = [1, 1, -1, -1]
        // E[x] = 0, E[y] = 0, E[xy] = (1 + (-1) + (-1) + 1)/4 = 0
        // Cov(x,y) = 0
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::<f64>::new(4).init((&s,), i64::MIN);

        s.push(1, &[1.0, 1.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 1);
        s.push(2, &[-1.0, 1.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 2);
        s.push(3, &[1.0, -1.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 3);
        s.push(4, &[-1.0, -1.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 4);

        let cov = out.last().unwrap();
        assert!(
            (cov[1]).abs() < 1e-10,
            "Cov(x,y) should be 0, got {}",
            cov[1]
        );
    }

    #[test]
    fn cov_nan() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::<f64>::new(3).init((&s,), i64::MIN);

        s.push(1, &[1.0, 2.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 1);
        s.push(2, &[f64::NAN, 4.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 2);
        s.push(3, &[3.0, 6.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 3);

        let cov = out.last().unwrap();
        // Element 0 has NaN in window → Var(x) and Cov(x,y) should be NaN
        assert!(cov[0].is_nan()); // Var(x)
        assert!(cov[1].is_nan()); // Cov(x,y)
        assert!(cov[2].is_nan()); // Cov(y,x)
        // Var(y) should still be valid (no NaN in y)
        assert!(!cov[3].is_nan());
    }

    #[test]
    fn cov_nan_eviction() {
        // NaN exits window → valid covariance resumes.
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::<f64>::new(2).init((&s,), i64::MIN);

        s.push(1, &[f64::NAN, 1.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 1);
        // Var(x) NaN, Cov NaN, Var(y) valid
        let cov = out.last().unwrap();
        assert!(cov[0].is_nan());
        assert!(cov[1].is_nan());
        assert!(!cov[3].is_nan());

        s.push(2, &[2.0, 4.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 2);
        // NaN still in window for elem 0
        assert!(out.last().unwrap()[0].is_nan());

        s.push(3, &[4.0, 8.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 3);
        // Window [[2,4],[4,8]] → NaN evicted
        let cov = out.last().unwrap();
        assert!(!cov[0].is_nan(), "Var(x) should be valid after NaN eviction");
        // Var([2,4]) = (4+16)/2 - (6/2)² = 10 - 9 = 1
        assert!((cov[0] - 1.0).abs() < 1e-10, "Var(x) = {}", cov[0]);
        // Cov([2,4],[4,8]) = E[xy]-E[x]E[y] = (8+32)/2 - 3*6 = 20 - 18 = 2
        assert!((cov[1] - 2.0).abs() < 1e-10, "Cov(x,y) = {}", cov[1]);
        assert!((cov[2] - cov[1]).abs() < 1e-10); // symmetric
    }

    #[test]
    fn cov_nan_both_elements() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::<f64>::new(3).init((&s,), i64::MIN);

        s.push(1, &[1.0, 2.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 1);
        s.push(2, &[f64::NAN, f64::NAN]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 2);
        s.push(3, &[3.0, 6.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 3);

        // Both elements have NaN in window → entire matrix is NaN
        let cov = out.last().unwrap();
        for v in cov {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn cov_nan_selective_recovery() {
        // NaN only in element 0 at tick 2. After eviction, full matrix recovers.
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::<f64>::new(2).init((&s,), i64::MIN);

        s.push(1, &[1.0, 1.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 1);
        s.push(2, &[f64::NAN, 2.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 2);

        let cov = out.last().unwrap();
        assert!(cov[0].is_nan()); // Var(x)
        assert!(cov[1].is_nan()); // Cov(x,y) — x has NaN
        assert!(!cov[3].is_nan()); // Var(y) — y is clean

        s.push(3, &[3.0, 3.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 3);
        s.push(4, &[5.0, 5.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, 4);

        // Window [[3,3],[5,5]] — all clean
        let cov = out.last().unwrap();
        for v in cov {
            assert!(!v.is_nan(), "all entries should be valid after NaN eviction");
        }
        // Var([3,5]) = (9+25)/2 - (8/2)² = 17-16 = 1
        assert!((cov[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "1D input")]
    fn cov_rejects_2d() {
        let s = Series::<f64>::new(&[2, 3]);
        let _ = RollingCovariance::<f64>::new(5).init((&s,), i64::MIN);
    }
}
