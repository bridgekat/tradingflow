use num_traits::Float;

use super::Windowed;
use crate::data::Scalar;
use crate::graph::cb::Comp;
use crate::operators::rolling::RollingMean;
use crate::operators::structural::buffer;

/// Rolling mean of the last `n` ticks: `ma(n) @ x`. Self-recording;
/// `NaN` (un-notified) until `n` values have been seen.
pub fn ma<T: Scalar + Float, const N: usize>(n: usize) -> Windowed<T, N, RollingMean<T, N>> {
    Comp(buffer(n), RollingMean::count(n))
}
