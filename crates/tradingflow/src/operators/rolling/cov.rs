use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`cov`].
pub struct CovAccumulator<T: Scalar + Float> {
    /// Number of elements in the input vector.
    k: usize,
    /// `pair_count[i * k + j]` is the number of `(i, j)`-complete ticks.
    pair_count: Vec<usize>,
    /// `pair_sum[i * k + j]` is sum of `x[i]` over the `(i, j)`-complete ticks.
    pair_sum: Vec<T>,
    /// `pair_sum2[i * k + j]` is sum of `x[i] * x[j]` over the `(i, j)`-complete ticks.
    pair_sum2: Vec<T>,
    /// Minimum number of complete ticks required to produce an output.
    min_count: usize,
}

impl<T: Scalar + Float> CovAccumulator<T> {
    fn new(min_count: usize) -> Self {
        Self {
            k: 0,
            pair_count: Vec::new(),
            pair_sum: Vec::new(),
            pair_sum2: Vec::new(),
            min_count: min_count.max(2),
        }
    }
}

impl<T: Scalar + Float> Accumulator<T, 1, T, 2> for CovAccumulator<T> {
    fn init(&mut self, extents: [usize; 1]) -> Array<T, 2> {
        let k = extents[0];
        self.k = k;
        self.pair_count = vec![0; k * k];
        self.pair_sum = vec![T::zero(); k * k];
        self.pair_sum2 = vec![T::zero(); k * k];
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
                self.pair_count[i * k + j] += 1;
                self.pair_sum[i * k + j] = self.pair_sum[i * k + j] + xi;
                self.pair_sum2[i * k + j] = self.pair_sum2[i * k + j] + xi * xj;
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
                self.pair_count[i * k + j] -= 1;
                self.pair_sum[i * k + j] = self.pair_sum[i * k + j] - xi;
                self.pair_sum2[i * k + j] = self.pair_sum2[i * k + j] - xi * xj;
            }
        }
    }

    fn write(&mut self, out: &mut Array<T, 2>, _: SeriesView<'_, T, 1>) {
        let k = self.k;
        let out = out.data_mut();
        for i in 0..k {
            for j in 0..k {
                if self.pair_count[i * k + j] >= self.min_count {
                    let n = T::from(self.pair_count[i * k + j]).unwrap();
                    let mean_xi = self.pair_sum[i * k + j] / n;
                    let mean_xj = self.pair_sum[j * k + i] / n;
                    let cov = (self.pair_sum2[i * k + j] - n * mean_xi * mean_xj) / (n - T::one());
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
    Rolling::new(window.into(), CovAccumulator::new(min_count))
}

/// Pairwise rolling sample covariance matrix (the unbiased `n − 1` estimator;
/// array extents `[K] -> [K, K]`) over a specified window, ingesting one
/// sample per signal. Non-finite values are skipped pairwise-complete: result
/// element `[i, j]` is computed over ticks where both components are finite,
/// and needs at least 2 such ticks.
pub fn cov<T: Scalar + Float>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, 1>), Outputs = ArrayPort<T, 2>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_cov(window, min_count))
}
