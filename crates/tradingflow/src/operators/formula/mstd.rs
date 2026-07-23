use num_traits::Float;

use super::mvar::mvar;
use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::graph::cb::Comp;
use crate::operators::num::sqrt;
use crate::ports::ArrayPort;

/// Rolling standard deviation of the last `n` ticks (variance → square root,
/// fused): `mstd(n) @ x`. Self-recording.
pub fn mstd<T: Scalar + Float, const N: usize>(
    n: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Comp(mvar(n), sqrt())
}
