use num_traits::Float;

use super::base::{Accumulator, rolling, series_rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::Operator;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`var`].
pub struct VarAccumulator<T: Scalar + Float> {
    count: Vec<usize>,
    sum: Vec<T>,
    sum2: Vec<T>,
    min_count: usize,
}

impl<T: Scalar + Float> VarAccumulator<T> {
    fn new(min_count: usize) -> Self {
        Self {
            count: Vec::new(),
            sum: Vec::new(),
            sum2: Vec::new(),
            min_count: min_count.max(2),
        }
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for VarAccumulator<T> {
    /// Maintained purely through `add`/`remove`; never reads the window.
    const NEEDS_WINDOW: bool = false;

    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.count = vec![0; stride];
        self.sum = vec![T::zero(); stride];
        self.sum2 = vec![T::zero(); stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.count[j] += 1;
                self.sum[j] = self.sum[j] + x;
                self.sum2[j] = self.sum2[j] + x * x;
            }
        });
    }

    fn remove(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.count[j] -= 1;
                self.sum[j] = self.sum[j] - x;
                self.sum2[j] = self.sum2[j] - x * x;
            }
        });
    }

    fn write(&mut self, out: &mut Array<T, N>, _: SeriesView<'_, T, N>) {
        for (j, o) in out.data_mut().iter_mut().enumerate() {
            if self.count[j] >= self.min_count {
                let n = T::from(self.count[j]).unwrap();
                let mean = self.sum[j] / n;
                let var = (self.sum2[j] - n * mean * mean) / (n - T::one());
                *o = var.max(T::zero());
            } else {
                *o = T::nan();
            }
        }
    }
}

/// [`var`] over an explicitly recorded series.
pub fn series_var<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    series_rolling(window, VarAccumulator::new(min_count))
}

/// Elementwise rolling sample variance (the unbiased `n − 1` estimator) over
/// a specified window, ingesting one sample per signal. Non-finite values are
/// skipped.
pub fn var<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
    min_count: usize,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    rolling(window, VarAccumulator::new(min_count))
}
