use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`mean_linear`].
///
/// A sample's weight is its 1-based window position (newest heaviest), which
/// admits an `O(1)` sliding update: a sample enters with weight equal to the
/// row count at insertion, and every row eviction lowers all remaining weights
/// by one — `wsum -= sum` and `wtot -= count`, no position bookkeeping.
pub struct MeanLinearAccumulator<T: Scalar + Float> {
    rows: usize,
    count: Vec<usize>,
    sum: Vec<T>,
    wsum: Vec<T>,
    sum_w: Vec<T>,
    min_count: usize,
}

impl<T: Scalar + Float> MeanLinearAccumulator<T> {
    fn new(min_count: usize) -> Self {
        Self {
            rows: 0,
            count: Vec::new(),
            sum: Vec::new(),
            wsum: Vec::new(),
            sum_w: Vec::new(),
            min_count: min_count.max(1),
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for MeanLinearAccumulator<T> {
    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.rows = 0;
        self.count = vec![0; stride];
        self.sum = vec![T::zero(); stride];
        self.wsum = vec![T::zero(); stride];
        self.sum_w = vec![T::zero(); stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, a: ArrayView<T, N>) {
        self.rows += 1;
        let w = T::from(self.rows).unwrap();
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.count[j] += 1;
                self.sum[j] = self.sum[j] + x;
                self.wsum[j] = self.wsum[j] + w * x;
                self.sum_w[j] = self.sum_w[j] + w;
            }
        });
    }

    fn remove(&mut self, a: ArrayView<T, N>) {
        // The evicted row is the oldest: weight 1.
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.count[j] -= 1;
                self.sum[j] = self.sum[j] - x;
                self.wsum[j] = self.wsum[j] - x;
                self.sum_w[j] = self.sum_w[j] - T::one();
            }
        });
        // Every remaining sample shifts down one position.
        for j in 0..self.count.len() {
            self.wsum[j] = self.wsum[j] - self.sum[j];
            self.sum_w[j] = self.sum_w[j] - T::from(self.count[j]).unwrap();
        }
        self.rows -= 1;
    }

    fn write(&mut self, out: &mut Array<T, N>, _: SeriesView<'_, T, N>) {
        for (j, o) in out.data_mut().iter_mut().enumerate() {
            *o = if self.count[j] >= self.min_count {
                self.wsum[j] / self.sum_w[j]
            } else {
                T::nan()
            };
        }
    }
}

/// [`mean_linear`] over an explicitly recorded series.
pub fn series_mean_linear<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Rolling::new(window.into(), MeanLinearAccumulator::new(min_count))
}

/// Elementwise linearly-weighted rolling mean (WMA) over a specified window,
/// ingesting one sample per signal: weights grow linearly with window
/// position, newest heaviest. Non-finite values are skipped (contributing to
/// neither the numerator nor the weight sum).
pub fn mean_linear<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_mean_linear(window, min_count))
}
