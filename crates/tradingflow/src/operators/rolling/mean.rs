use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`mean`].
pub struct MeanAccumulator<T: Scalar + Float> {
    count: Vec<usize>,
    sum: Vec<T>,
    min_count: usize,
}

impl<T: Scalar + Float> MeanAccumulator<T> {
    fn new(min_count: usize) -> Self {
        Self {
            count: Vec::new(),
            sum: Vec::new(),
            min_count: min_count.max(1),
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for MeanAccumulator<T> {
    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.count = vec![0; stride];
        self.sum = vec![T::zero(); stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.count[j] += 1;
                self.sum[j] = self.sum[j] + x;
            }
        });
    }

    fn remove(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.count[j] -= 1;
                self.sum[j] = self.sum[j] - x;
            }
        });
    }

    fn write(&mut self, out: &mut Array<T, N>, _: SeriesView<'_, T, N>) {
        for (j, o) in out.data_mut().iter_mut().enumerate() {
            if self.count[j] >= self.min_count {
                let n = T::from(self.count[j]).unwrap();
                *o = self.sum[j] / n;
            } else {
                *o = T::nan();
            }
        }
    }
}

/// [`mean`] over an explicitly recorded series.
pub fn series_mean<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Rolling::new(window.into(), MeanAccumulator::new(min_count))
}

/// Elementwise rolling mean over a specified window, ingesting one sample per
/// signal. Non-finite values are skipped.
pub fn mean<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_mean(window, min_count))
}
