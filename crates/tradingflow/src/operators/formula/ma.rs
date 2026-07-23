use num_traits::Float;

use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::graph::cb::Comp;
use crate::operators::rolling::RollingMean;
use crate::operators::series::buffer_n;
use crate::ports::ArrayPort;

/// Rolling mean of the last `n` ticks: `ma(n) @ x`. Self-recording;
/// `NaN` (un-notified) until `n` values have been seen.
pub fn ma<T: Scalar + Float, const N: usize>(
    n: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Comp(buffer_n(n), RollingMean::count(n))
}
