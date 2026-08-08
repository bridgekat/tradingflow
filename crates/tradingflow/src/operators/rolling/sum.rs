use num_traits::Float;

use super::base::{Accumulator, rolling, series_rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::Operator;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`sum`].
pub struct SumAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    count: Vec<usize>,
    min_count: usize,
}

impl<T: Scalar + Float> SumAccumulator<T> {
    fn new(min_count: usize) -> Self {
        Self {
            sum: Vec::new(),
            count: Vec::new(),
            min_count: min_count.max(1),
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for SumAccumulator<T> {
    /// Maintained purely through `add`/`remove`; never reads the window.
    const NEEDS_WINDOW: bool = false;

    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.sum = vec![T::zero(); stride];
        self.count = vec![0; stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.sum[j] = self.sum[j] + x;
                self.count[j] += 1;
            }
        });
    }

    fn remove(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.sum[j] = self.sum[j] - x;
                self.count[j] -= 1;
            }
        });
    }

    fn write(&mut self, out: &mut Array<T, N>, _: SeriesView<'_, T, N>) {
        for (j, o) in out.data_mut().iter_mut().enumerate() {
            if self.count[j] >= self.min_count {
                *o = self.sum[j];
            } else {
                *o = T::nan();
            }
        }
    }
}

/// [`sum`] over an explicitly recorded series.
pub fn series_sum<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    series_rolling(window, SumAccumulator::new(min_count))
}

/// Elementwise rolling sum over a specified window, ingesting one sample per
/// signal. Non-finite values are skipped.
pub fn sum<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    rolling(window, SumAccumulator::new(min_count))
}
