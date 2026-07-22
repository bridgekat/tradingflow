use num_traits::Float;

use super::Windowed;
use crate::data::Scalar;
use crate::graph::cb::Comp;
use crate::operators::rolling::RollingSum;
use crate::operators::structural::buffer;

/// Rolling sum of the last `n` ticks: `msum(n) @ x`. Self-recording.
pub fn msum<T: Scalar + Float, const N: usize>(n: usize) -> Windowed<T, N, RollingSum<T, N>> {
    Comp(buffer(n), RollingSum::count(n))
}
