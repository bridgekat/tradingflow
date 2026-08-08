use num_traits::Float;

use super::base::{Accumulator, rolling, series_rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::Operator;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`mean_exp`].
pub struct MeanExpAccumulator<T: Scalar + Float> {
    alpha: T,
    one_minus_alpha: T,
    count: Vec<usize>,
    sum: Vec<T>,
    sum_w: Vec<T>,
    min_count: usize,
}

impl<T: Scalar + Float> MeanExpAccumulator<T> {
    fn new(span: usize, min_count: usize) -> Self {
        assert!(span >= 1, "mean_exp: span must be at least 1");
        let span = T::from(span).unwrap();
        let alpha = T::from(2.0).unwrap() / (span + T::one());
        Self {
            alpha,
            one_minus_alpha: T::one() - alpha,
            count: Vec::new(),
            sum: Vec::new(),
            sum_w: Vec::new(),
            min_count: min_count.max(1),
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for MeanExpAccumulator<T> {
    const NEEDS_WINDOW: bool = false;

    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.sum = vec![T::zero(); stride];
        self.sum_w = vec![T::zero(); stride];
        self.count = vec![0; stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            self.sum[j] = self.sum[j] * self.one_minus_alpha;
            self.sum_w[j] = self.sum_w[j] * self.one_minus_alpha;
            if x.is_finite() {
                self.count[j] += 1;
                self.sum[j] = self.sum[j] + self.alpha * x;
                self.sum_w[j] = self.sum_w[j] + self.alpha;
            }
        });
    }

    fn remove(&mut self, _: ArrayView<T, N>) {
        unreachable!("mean_exp: window is always unbounded")
    }

    fn write(&mut self, out: &mut Array<T, N>, _: SeriesView<'_, T, N>) {
        for (j, o) in out.data_mut().iter_mut().enumerate() {
            if self.count[j] >= self.min_count && self.sum_w[j] > T::zero() {
                *o = self.sum[j] / self.sum_w[j];
            } else {
                *o = T::nan();
            }
        }
    }
}

/// [`mean_exp`] over an explicitly recorded series.
pub fn series_mean_exp<T: Scalar + Float, const N: usize>(
    span: usize,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let window = Retention::unbounded();
    series_rolling(window, MeanExpAccumulator::<T>::new(span, min_count))
}

/// Elementwise exponentially-weighted rolling mean (EMA), ingesting one sample
/// per signal. Non-finite values are skipped, but still age the average.
///
/// `span` is the conventional EWMA span, so the decay is
/// `alpha = 2 / (span + 1)`: a `span` of 1 collapses to the latest sample.
pub fn mean_exp<T: Scalar + Float, const N: usize>(
    span: usize,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = Retention::unbounded();
    rolling(window, MeanExpAccumulator::<T>::new(span, min_count))
}
