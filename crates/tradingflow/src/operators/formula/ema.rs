use num_traits::Float;

use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::graph::cb::Comp;
use crate::operators::rolling::Ema;
use crate::operators::series::buffer_n;
use crate::ports::ArrayPort;

/// Exponential moving average with window-normalized weights (see [`Ema`]):
/// `ema(alpha, window) @ x`. Self-recording.
pub fn ema<T: Scalar + Float, const N: usize>(
    alpha: T,
    n: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Comp(buffer_n(n), Ema::new(alpha, n))
}
