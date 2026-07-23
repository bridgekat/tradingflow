use num_traits::Float;

use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::graph::cb::Comp;
use crate::operators::rolling::RollingVariance;
use crate::operators::series::buffer_n;
use crate::ports::ArrayPort;

/// Rolling population variance of the last `n` ticks: `mvar(n) @ x`.
/// Self-recording.
pub fn mvar<T: Scalar + Float, const N: usize>(
    n: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Comp(buffer_n(n), RollingVariance::count(n))
}
