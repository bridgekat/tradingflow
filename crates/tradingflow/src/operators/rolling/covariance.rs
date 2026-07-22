use num_traits::Float;

use crate::data::Scalar;

use super::{Accumulator, Rolling, Window, rolling};

/// Incremental pairwise covariance matrix (`[K] → [K, K]`).
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

    #[expect(
        clippy::needless_range_loop,
        reason = "i/j also address the flat k*k `sum_cross` cross-moment matrix"
    )]
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

    #[expect(
        clippy::needless_range_loop,
        reason = "i/j also address the flat k*k `sum_cross` cross-moment matrix"
    )]
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

/// Pairwise rolling covariance matrix (`[K] → [K, K]`, output rank 2).
pub type RollingCovariance<T> = Rolling<CovarianceAccumulator<T>, 1, 2>;

/// Pairwise rolling covariance matrix (`[K] → [K, K]`) over a recorded
/// [`Series`](tradingflow_data::Series).
pub fn rolling_covariance<T: Scalar + Float>(window: Window) -> RollingCovariance<T> {
    rolling(window)
}
