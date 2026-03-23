//! Rolling covariance matrix operator.

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

/// State: window size and K (number of elements).
pub struct CovState {
    window: usize,
    k: usize,
}

impl<T: Scalar + Float> Operator for RollingCovariance<T> {
    type State = CovState;
    type Inputs = (Series<T>,);
    type Output = Series<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (CovState, Series<T>) {
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
        };
        (state, Series::new(&[k, k]))
    }

    fn compute(
        state: &mut CovState,
        inputs: (&Series<T>,),
        output: &mut Series<T>,
        timestamp: i64,
    ) -> bool {
        let series = inputs.0;
        let len = series.len();
        let k = state.k;
        let start = len.saturating_sub(state.window);
        let count = T::from(len - start).unwrap();

        // Accumulate sums and cross-products
        let mut sum = vec![T::zero(); k];
        let mut has_nan = vec![false; k];

        // sum_cross[i * k + j] = sum(x_i * x_j)
        let mut sum_cross = vec![T::zero(); k * k];

        for t in start..len {
            let row = series.at(t);
            for i in 0..k {
                let xi = row[i];
                if xi.is_nan() {
                    has_nan[i] = true;
                } else if !has_nan[i] {
                    sum[i] = sum[i] + xi;
                }
                for j in i..k {
                    let xj = row[j];
                    if !xi.is_nan() && !xj.is_nan() && !has_nan[i] && !has_nan[j] {
                        let prod = xi * xj;
                        sum_cross[i * k + j] = sum_cross[i * k + j] + prod;
                        if i != j {
                            sum_cross[j * k + i] = sum_cross[j * k + i] + prod;
                        }
                    }
                }
            }
        }

        let mut buf = vec![T::nan(); k * k];
        for i in 0..k {
            for j in 0..k {
                if !has_nan[i] && !has_nan[j] {
                    let mean_i = sum[i] / count;
                    let mean_j = sum[j] / count;
                    buf[i * k + j] = sum_cross[i * k + j] / count - mean_i * mean_j;
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
    #[should_panic(expected = "1D input")]
    fn cov_rejects_2d() {
        let s = Series::<f64>::new(&[2, 3]);
        let _ = RollingCovariance::<f64>::new(5).init((&s,), i64::MIN);
    }
}
