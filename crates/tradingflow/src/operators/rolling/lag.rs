use num_traits::Float;

use super::base::{Accumulator, Rolling};
use crate::data::{Array, ArrayView, Instant, Retention, Scalar, array};
use crate::fuse;
use crate::graph::{Operator, OperatorExt};
use crate::operators::elem::{div, sub};
use crate::operators::series::buffer;
use crate::ports::{ArrayPort, SeriesPort, SignalPort};

/// Accumulator for [`lag`].
pub struct LagAccumulator<T: Scalar + Float> {
    value: Vec<T>,
}

impl<T: Scalar + Float> LagAccumulator<T> {
    fn new() -> Self {
        Self { value: Vec::new() }
    }
}

impl Default for LagAccumulator<f64> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float, const N: usize> Accumulator<T, N, T, N> for LagAccumulator<T> {
    fn init(&mut self, extents: [usize; N]) -> Array<T, N> {
        let stride = extents.iter().product();
        self.value = vec![T::nan(); stride];
        Array::full(extents, T::nan())
    }

    fn add(&mut self, _: ArrayView<T, N>) {}

    fn remove(&mut self, a: ArrayView<T, N>) {
        array::for_each(a, |j, &x| {
            self.value[j] = x;
        });
    }

    fn write(&mut self, out: &mut Array<T, N>, _: usize) {
        out.data_mut().clone_from_slice(&self.value);
    }
}

/// [`lag`] over an explicitly recorded series.
pub fn series_lag<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
) -> impl Operator<Inputs = SeriesPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Rolling::new(window.into(), LagAccumulator::new())
}

/// The last sample before the specified window.
/// Returns `NaN` if there is no such sample.
pub fn lag<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    let window = window.into();
    buffer(window).then(series_lag(window))
}

/// Change relative to the last sample before the specified window.
/// Returns `NaN` if there is no such sample.
pub fn diff<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    fuse!(@[crate::graph::cb] |c: SignalPort<0>, x: ArrayPort<T, N>| -> ArrayPort<T, N> {
        let prev = lag(window) @ (c, x);
        sub() @ (x, prev)
    })
}

/// Percentage change relative to the last sample before the specified window.
/// Returns `NaN` if there is no such sample.
pub fn pct_change<T: Scalar + Float, const N: usize>(
    window: impl Into<Retention>,
) -> impl Operator<Inputs = (SignalPort<0>, ArrayPort<T, N>), Outputs = ArrayPort<T, N>, Context = Instant>
{
    fuse!(@[crate::graph::cb] |c: SignalPort<0>, x: ArrayPort<T, N>| -> ArrayPort<T, N> {
        let prev = lag(window) @ (c, x);
        div() @ (sub() @ (x, prev), prev)
    })
}
