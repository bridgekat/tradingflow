//! Rolling covariance matrix accumulator.
//!
//! O(K²) per tick via incremental cross-product sums with non-finite
//! counting.

use num_traits::Float;


use crate::Scalar;

use super::accumulator::Accumulator;

/// Incremental pairwise covariance matrix accumulator.
///
/// Input must be 1D with shape `[K]`. Output has shape `[K, K]`.
///
/// `Cov(i,j) = E[x_i · x_j] − E[x_i] · E[x_j]` over the window.
/// Non-finite values (NaN, ±inf) are skipped and counted separately rather
/// than added to the running sums, since `inf − inf` would corrupt the
/// sums to NaN on eviction.  If any value in the window is non-finite for
/// either element `i` or `j`, the output `Cov(i,j)` is NaN.
pub struct CovarianceAccumulator<T: Scalar + Float> {
    k: usize,
    sum: Vec<T>,
    sum_cross: Vec<T>,
    nonfinite_count: Vec<u32>,
}

impl<T: Scalar + Float> Accumulator for CovarianceAccumulator<T> {
    type Scalar = T;

    fn new(input_shape: &[usize]) -> Self {
        assert_eq!(
            input_shape.len(),
            1,
            "CovarianceAccumulator requires 1D input, got shape {input_shape:?}",
        );
        let k = input_shape[0];
        Self {
            k,
            sum: vec![T::zero(); k],
            sum_cross: vec![T::zero(); k * k],
            nonfinite_count: vec![0; k],
        }
    }

    fn output_shape(input_shape: &[usize]) -> Vec<usize> {
        assert_eq!(
            input_shape.len(),
            1,
            "CovarianceAccumulator requires 1D input, got shape {input_shape:?}",
        );
        vec![input_shape[0], input_shape[0]]
    }

    fn add(&mut self, element: &[T]) {
        let k = self.k;
        for i in 0..k {
            let xi = element[i];
            if !xi.is_finite() {
                self.nonfinite_count[i] += 1;
            } else {
                self.sum[i] = self.sum[i] + xi;
            }
        }
        for i in 0..k {
            let xi = element[i];
            if !xi.is_finite() {
                continue;
            }
            for j in i..k {
                let xj = element[j];
                if !xj.is_finite() {
                    continue;
                }
                let prod = xi * xj;
                self.sum_cross[i * k + j] = self.sum_cross[i * k + j] + prod;
                if i != j {
                    self.sum_cross[j * k + i] = self.sum_cross[j * k + i] + prod;
                }
            }
        }
    }

    fn remove(&mut self, element: &[T]) {
        let k = self.k;
        for i in 0..k {
            let xi = element[i];
            if !xi.is_finite() {
                self.nonfinite_count[i] -= 1;
            } else {
                self.sum[i] = self.sum[i] - xi;
            }
        }
        for i in 0..k {
            let xi = element[i];
            if !xi.is_finite() {
                continue;
            }
            for j in i..k {
                let xj = element[j];
                if !xj.is_finite() {
                    continue;
                }
                let prod = xi * xj;
                self.sum_cross[i * k + j] = self.sum_cross[i * k + j] - prod;
                if i != j {
                    self.sum_cross[j * k + i] = self.sum_cross[j * k + i] - prod;
                }
            }
        }
    }

    fn write(&self, count: usize, output: &mut [T]) {
        let k = self.k;
        let n = T::from(count).unwrap();
        for i in 0..k {
            for j in 0..k {
                output[i * k + j] = if self.nonfinite_count[i] == 0 && self.nonfinite_count[j] == 0
                {
                    self.sum_cross[i * k + j] / n - (self.sum[i] / n) * (self.sum[j] / n)
                } else {
                    T::nan()
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::rolling::accumulator::Rolling;
    use crate::data::{Duration, Instant};
    use crate::{Notify, Operator, Series};

    type RollingCovariance = Rolling<CovarianceAccumulator<f64>>;

    fn ts(n: i64) -> Instant { Instant::from_nanos(n) }

    #[test]
    fn cov_basic() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::count(3).init((&s,), Instant::MIN);

        assert_eq!(out.shape(), &[2, 2]);

        s.push(ts(1), &[1.0, 2.0]);
        assert!(!RollingCovariance::compute(
            &mut state,
            (&s,),
            &mut out,
            ts(1),
            &Notify::new(&[], 0)
        ));

        s.push(ts(2), &[2.0, 4.0]);
        assert!(!RollingCovariance::compute(
            &mut state,
            (&s,),
            &mut out,
            ts(2),
            &Notify::new(&[], 0)
        ));

        s.push(ts(3), &[3.0, 6.0]);
        assert!(RollingCovariance::compute(
            &mut state,
            (&s,),
            &mut out,
            ts(3),
            &Notify::new(&[], 0)
        ));

        // Perfect linear correlation: y = 2x.
        // Var(x) = Var([1,2,3]) = 2/3. Cov(x,y) = 2 * Var(x) = 4/3.
        let cov = out.as_slice();
        assert!((cov[0] - 2.0 / 3.0).abs() < 1e-10); // Var(x)
        assert!((cov[1] - 4.0 / 3.0).abs() < 1e-10); // Cov(x,y)
        assert!((cov[2] - 4.0 / 3.0).abs() < 1e-10); // Cov(y,x)
        assert!((cov[3] - 8.0 / 3.0).abs() < 1e-10); // Var(y)
    }

    #[test]
    #[should_panic(expected = "requires 1D")]
    fn cov_rejects_scalar() {
        let s = Series::<f64>::new(&[]);
        RollingCovariance::count(3).init((&s,), Instant::MIN);
    }

    #[test]
    fn cov_nan_propagation() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::count(2).init((&s,), Instant::MIN);

        s.push(ts(1), &[f64::NAN, 1.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, ts(1), &Notify::new(&[], 0));

        s.push(ts(2), &[2.0, 2.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, ts(2), &Notify::new(&[], 0));

        let cov = out.as_slice();
        assert!(cov[0].is_nan()); // Var(x): NaN in x
        assert!(cov[1].is_nan()); // Cov(x,y): NaN in x
        assert!(cov[2].is_nan()); // Cov(y,x): NaN in x
        assert!(!cov[3].is_nan()); // Var(y): no NaN in y
    }

    #[test]
    fn cov_time_delta() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = RollingCovariance::time_delta(Duration::from_nanos(200)).init((&s,), Instant::MIN);

        s.push(ts(100), &[1.0, 2.0]);
        assert!(RollingCovariance::compute(
            &mut state,
            (&s,),
            &mut out,
            ts(100),
            &Notify::new(&[], 0)
        ));
        // Single element → all covariances = 0.
        assert_eq!(out.as_slice(), &[0.0, 0.0, 0.0, 0.0]);

        s.push(ts(200), &[3.0, 6.0]);
        RollingCovariance::compute(&mut state, (&s,), &mut out, ts(200), &Notify::new(&[], 0));

        // Cov([1,3], [2,6]): Var(x)=1, Cov(x,y)=2, Var(y)=4.
        let cov = out.as_slice();
        assert!((cov[0] - 1.0).abs() < 1e-10);
        assert!((cov[1] - 2.0).abs() < 1e-10);
        assert!((cov[3] - 4.0).abs() < 1e-10);
    }
}
