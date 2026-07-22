use num_traits::Float;

use super::Windowed;
use crate::data::Scalar;
use crate::graph::cb::Comp;
use crate::operators::rolling::RollingVariance;
use crate::operators::structural::buffer;

/// Rolling population variance of the last `n` ticks: `mvar(n) @ x`.
/// Self-recording.
pub fn mvar<T: Scalar + Float, const N: usize>(n: usize) -> Windowed<T, N, RollingVariance<T, N>> {
    Comp(buffer(n), RollingVariance::count(n))
}
