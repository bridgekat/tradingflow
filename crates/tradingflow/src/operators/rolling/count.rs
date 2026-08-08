use num_traits::Float;

use super::base::{Accumulator, rolling, series_rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, SeriesView, array};
use crate::graph::Operator;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`count`].
pub struct CountAccumulator {
    count: Vec<usize>,
}

impl CountAccumulator {
    pub fn new() -> Self {
        Self { count: Vec::new() }
    }
}

impl Default for CountAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for CountAccumulator {
    const NEEDS_WINDOW: bool = false;

    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.count = vec![0; stride];
        Array::full(extents, T::zero())
    }

    fn add(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.count[j] += 1;
            }
        });
    }

    fn remove(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            if x.is_finite() {
                self.count[j] -= 1;
            }
        });
    }

    fn write(&mut self, out: &mut Array<T, N>, _: SeriesView<'_, T, N>) {
        for (j, o) in out.data_mut().iter_mut().enumerate() {
            *o = T::from(self.count[j]).unwrap();
        }
    }
}

/// [`count`] over an explicitly recorded series.
pub fn series_count<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    series_rolling(window, CountAccumulator::new())
}

/// Elementwise count of the finite samples in a specified window, ingesting
/// one sample per signal. Always defined (zero when the window holds no finite
/// samples).
pub fn count<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    rolling(window, CountAccumulator::new())
}
