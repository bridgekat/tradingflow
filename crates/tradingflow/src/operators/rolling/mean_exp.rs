use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, array};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`mean_exp`].
pub struct MeanExpAccumulator<T: Scalar + Float> {
    alpha: T,
    one_minus_alpha: T,
    added: usize,
    removed: usize,
    sum: Vec<T>,
    sum_w: Vec<T>,
    count: Vec<usize>,
    min_count: usize,
}

impl<T: Scalar + Float> MeanExpAccumulator<T> {
    fn new(alpha: T, min_count: usize) -> Self {
        assert!(
            alpha > T::zero() && alpha <= T::one(),
            "alpha must be in (0, 1]"
        );
        assert!(min_count > 0, "min_count must be positive");
        Self {
            alpha,
            one_minus_alpha: T::one() - alpha,
            added: 0,
            removed: 0,
            sum: Vec::new(),
            sum_w: Vec::new(),
            count: Vec::new(),
            min_count,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for MeanExpAccumulator<T> {
    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.sum = vec![T::zero(); stride];
        self.sum_w = vec![T::zero(); stride];
        self.count = vec![0; stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, a: ArrayView<T, N>) {
        self.added += 1;
        array::for_each(a, |j, &x| {
            self.sum[j] = self.sum[j] * self.one_minus_alpha;
            self.sum_w[j] = self.sum_w[j] * self.one_minus_alpha;
            if x.is_finite() {
                self.sum[j] = self.sum[j] + self.alpha * x;
                self.sum_w[j] = self.sum_w[j] + self.alpha;
                self.count[j] += 1;
            }
        });
    }

    fn remove(&mut self, a: ArrayView<T, N>) {
        let age = i32::try_from(self.added - self.removed - 1).unwrap_or(i32::MAX);
        let w = self.alpha * self.one_minus_alpha.powi(age);
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.sum[j] = self.sum[j] - w * x;
                self.sum_w[j] = self.sum_w[j] - w;
                self.count[j] -= 1;
            }
        });
        self.removed += 1;
    }

    fn write(&mut self, out: &mut Array<T, N>, _: usize) {
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
    alpha: T,
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Rolling::new(window.into(), MeanExpAccumulator::new(alpha, min_count))
}

/// Elementwise exponentially-weighted rolling mean (EMA) over a specified
/// window, ingesting one sample per signal. Non-finite values are skipped.
pub fn mean_exp<T: Scalar + Float, const N: usize>(
    alpha: T,
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_mean_exp(alpha, window, min_count))
}
