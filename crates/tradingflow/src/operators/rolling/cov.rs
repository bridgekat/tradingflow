use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`cov`].
pub struct CovarianceAccumulator<T: Scalar + Float> {
    /// Number of elements in the input vector.
    k: usize,
    /// `pair_sum[i * k + j]` is sum of `x[i]` over the `(i, j)`-complete ticks.
    pair_sum: Vec<T>,
    /// `pair_cross_sum[i * k + j]` is sum of `x[i] * x[j]` over the `(i, j)`-complete ticks.
    pair_cross_sum: Vec<T>,
    /// `pair_count[i * k + j]` is the number of `(i, j)`-complete ticks.
    pair_count: Vec<usize>,
    /// Minimum number of complete ticks required to produce an output.
    min_count: usize,
}

impl<T: Scalar + Float> CovarianceAccumulator<T> {
    fn new(min_count: usize) -> Self {
        assert!(min_count > 0, "min_count must be positive");
        Self {
            k: 0,
            pair_sum: Vec::new(),
            pair_cross_sum: Vec::new(),
            pair_count: Vec::new(),
            min_count,
        }
    }
}

impl<T: Scalar + Float> Accumulator<T, 1, T, 2> for CovarianceAccumulator<T> {
    fn init(&mut self, extents: [usize; 1]) -> Array<T, 2> {
        let k = extents[0];
        self.k = k;
        self.pair_sum = vec![T::zero(); k * k];
        self.pair_cross_sum = vec![T::zero(); k * k];
        self.pair_count = vec![0; k * k];
        Array::full([k, k], T::nan())
    }

    fn add(&mut self, a: ArrayView<T, 1>) {
        let k = self.k;
        let a = a.to_contiguous();
        for i in 0..k {
            let xi = a[i];
            if !xi.is_finite() {
                continue;
            }
            for j in 0..k {
                let xj = a[j];
                if !xj.is_finite() {
                    continue;
                }
                self.pair_sum[i * k + j] = self.pair_sum[i * k + j] + xi;
                self.pair_cross_sum[i * k + j] = self.pair_cross_sum[i * k + j] + xi * xj;
                self.pair_count[i * k + j] += 1;
            }
        }
    }

    fn remove(&mut self, a: ArrayView<T, 1>) {
        let k = self.k;
        let a = a.to_contiguous();
        for i in 0..k {
            let xi = a[i];
            if !xi.is_finite() {
                continue;
            }
            for j in 0..k {
                let xj = a[j];
                if !xj.is_finite() {
                    continue;
                }
                self.pair_sum[i * k + j] = self.pair_sum[i * k + j] - xi;
                self.pair_cross_sum[i * k + j] = self.pair_cross_sum[i * k + j] - xi * xj;
                self.pair_count[i * k + j] -= 1;
            }
        }
    }

    fn write(&mut self, out: &mut Array<T, 2>, _: usize) {
        let k = self.k;
        let out = out.data_mut();
        for i in 0..k {
            for j in 0..k {
                if self.pair_count[i * k + j] >= self.min_count {
                    let n = T::from(self.pair_count[i * k + j]).unwrap();
                    let mean_xi = self.pair_sum[i * k + j] / n;
                    let mean_xj = self.pair_sum[j * k + i] / n;
                    let cov = self.pair_cross_sum[i * k + j] / n - mean_xi * mean_xj;
                    out[i * k + j] = if i == j { cov.max(T::zero()) } else { cov };
                } else {
                    out[i * k + j] = T::nan();
                }
            }
        }
    }
}

/// [`cov`] over an explicitly recorded series.
pub fn series_cov<T: Scalar + Float>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, 1>, Outputs = ArrayPort<T, 2>, Context = Instant> {
    Rolling::new(window.into(), CovarianceAccumulator::new(min_count))
}

/// Pairwise rolling covariance matrix (array extents `[K] -> [K, K]`) over a
/// specified window, ingesting one sample per signal. Non-finite values
/// are skipped pairwise-complete: result element `[i, j]` is computed over
/// ticks where both components are finite.
pub fn cov<T: Scalar + Float>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 1>), Outputs = ArrayPort<T, 2>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_cov(window, min_count))
}
