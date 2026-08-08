use num_traits::Float;

use super::base::{Accumulator, rolling, series_rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::Operator;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`kurt`].
pub struct KurtAccumulator<T: Scalar + Float> {
    count: Vec<usize>,
    sum: Vec<T>,
    sum2: Vec<T>,
    sum3: Vec<T>,
    sum4: Vec<T>,
    min_count: usize,
}

impl<T: Scalar + Float> KurtAccumulator<T> {
    fn new(min_count: usize) -> Self {
        Self {
            count: Vec::new(),
            sum: Vec::new(),
            sum2: Vec::new(),
            sum3: Vec::new(),
            sum4: Vec::new(),
            min_count: min_count.max(4),
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for KurtAccumulator<T> {
    const NEEDS_WINDOW: bool = false;

    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.count = vec![0; stride];
        self.sum = vec![T::zero(); stride];
        self.sum2 = vec![T::zero(); stride];
        self.sum3 = vec![T::zero(); stride];
        self.sum4 = vec![T::zero(); stride];
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
                self.sum4[j] = self.sum4[j] + x2 * x2;
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
                self.sum4[j] = self.sum4[j] - x2 * x2;
            }
        });
    }

    fn write(&mut self, out: &mut Array<T, N>, _: SeriesView<'_, T, N>) {
        for (j, o) in out.data_mut().iter_mut().enumerate() {
            if self.count[j] >= self.min_count {
                let n = T::from(self.count[j]).unwrap();
                let mean = self.sum[j] / n;
                let mean2 = mean * mean;
                let (r2, r3, r4) = (self.sum2[j] / n, self.sum3[j] / n, self.sum4[j] / n);
                let (two, three) = (T::from(2).unwrap(), T::from(3).unwrap());
                let m2 = r2 - mean2;
                let m4 =
                    r4 - two * two * mean * r3 + two * three * mean2 * r2 - three * mean2 * mean2;
                if m2 > T::zero() {
                    let one = T::one();
                    let (two, three, six) = (one + one, T::from(3).unwrap(), T::from(6).unwrap());
                    let g2 = m4 / (m2 * m2) - three;
                    *o = ((n + one) * g2 + six) * (n - one) / ((n - two) * (n - three))
                } else {
                    *o = T::nan()
                }
            } else {
                *o = T::nan();
            }
        }
    }
}

/// [`kurt`] over an explicitly recorded series.
pub fn series_kurt<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    series_rolling(window, KurtAccumulator::new(min_count))
}

/// Elementwise rolling sample excess kurtosis (the adjusted estimator, as in
/// `pandas.kurt`) over a specified window, ingesting one sample per signal.
/// Non-finite values are skipped.
pub fn kurt<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    rolling(window, KurtAccumulator::new(min_count))
}
