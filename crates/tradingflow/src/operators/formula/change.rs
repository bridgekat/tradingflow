use num_traits::Float;

use super::lag::lag;
use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::graph::cb::{Comp, Fork, Id};
use crate::operators::elem;
use crate::ports::ArrayPort;

/// `n`-tick change `x − x₋ₙ` (the momentum shape): `change(n) @ x`.
/// Self-recording; `NaN` until the lag is available. The one-tick special
/// case, without a private record, is [`diff`](crate::operators::num::diff).
pub fn change<T: Scalar + Float, const N: usize>(
    n: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Comp(Fork(Id::default(), lag(n)), elem::subtract())
}
