use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, array};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`sum`].
pub struct SumAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    count: Vec<usize>,
    min_count: usize,
}

impl<T: Scalar + Float> SumAccumulator<T> {
    fn new(min_count: usize) -> Self {
        assert!(min_count > 0, "min_count must be positive");
        Self {
            sum: Vec::new(),
            count: Vec::new(),
            min_count,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for SumAccumulator<T> {
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

    fn write(&mut self, out: &mut Array<T, N>, _: usize) {
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
    Rolling::new(window.into(), SumAccumulator::new(min_count))
}

/// Elementwise rolling sum over a specified window, ingesting one sample per
/// signal. Non-finite values are skipped.
pub fn sum<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_sum(window, min_count))
}
