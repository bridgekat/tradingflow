use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::{Operator, OperatorExt};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`skew`].
pub struct SkewAccumulator<T: Scalar + Float> {
    count: Vec<usize>,
    sum: Vec<T>,
    sum2: Vec<T>,
    sum3: Vec<T>,
    min_count: usize,
}

impl<T: Scalar + Float> SkewAccumulator<T> {
    fn new(min_count: usize) -> Self {
        Self {
            count: Vec::new(),
            sum: Vec::new(),
            sum2: Vec::new(),
            sum3: Vec::new(),
            min_count: min_count.max(3),
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for SkewAccumulator<T> {
    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.count = vec![0; stride];
        self.sum = vec![T::zero(); stride];
        self.sum2 = vec![T::zero(); stride];
        self.sum3 = vec![T::zero(); stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                let x2 = x * x;
                self.count[j] += 1;
                self.sum[j] = self.sum[j] + x;
                self.sum2[j] = self.sum2[j] + x2;
                self.sum3[j] = self.sum3[j] + x2 * x;
            }
        });
    }

    fn remove(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                let x2 = x * x;
                self.count[j] -= 1;
                self.sum[j] = self.sum[j] - x;
                self.sum2[j] = self.sum2[j] - x2;
                self.sum3[j] = self.sum3[j] - x2 * x;
            }
        });
    }

    fn write(&mut self, out: &mut Array<T, N>, _: SeriesView<'_, T, N>) {
        for (j, o) in out.data_mut().iter_mut().enumerate() {
            if self.count[j] >= self.min_count {
                let n = T::from(self.count[j]).unwrap();
                let mean = self.sum[j] / n;
                let mean2 = mean * mean;
                let (r2, r3) = (self.sum2[j] / n, self.sum3[j] / n);
                let (two, three) = (T::from(2).unwrap(), T::from(3).unwrap());
                let m2 = r2 - mean2;
                let m3 = r3 - three * mean * r2 + two * mean * mean2;
                if m2 > T::zero() {
                    let one = T::one();
                    let g1 = m3 / m2.powf(T::from(1.5).unwrap());
                    *o = g1 * (n * (n - one)).sqrt() / (n - one - one);
                } else {
                    *o = T::nan();
                }
            } else {
                *o = T::nan();
            }
        }
    }
}

/// [`skew`] over an explicitly recorded series.
pub fn series_skew<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Rolling::new(window.into(), SkewAccumulator::new(min_count))
}

/// Elementwise rolling sample skewness (the adjusted Fisher–Pearson estimator,
/// as in `pandas.skew`) over a specified window, ingesting one sample per
/// signal. Non-finite values are skipped.
pub fn skew<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_skew(window, min_count))
}
