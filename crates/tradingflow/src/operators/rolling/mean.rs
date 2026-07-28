use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, array};
use crate::graph::{Segment, SegmentExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, ClockPort, SeriesPort};

/// Accumulator for [`mean`].
pub struct MeanAccumulator<T: Scalar + Float> {
    sum: Vec<T>,
    count: Vec<usize>,
    min_count: usize,
}

impl<T: Scalar + Float> MeanAccumulator<T> {
    fn new(min_count: usize) -> Self {
        assert!(min_count > 0, "min_count must be positive");
        Self {
            sum: Vec::new(),
            count: Vec::new(),
            min_count,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for MeanAccumulator<T> {
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
) -> impl Segment<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Rolling::new(window.into(), MeanAccumulator::new(min_count))
}

/// Elementwise rolling mean over a specified window, ingesting one sample per
/// clock signal. Non-finite values are skipped.
pub fn mean<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Segment<Inputs = (ClockPort, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_mean(window, min_count))
}
