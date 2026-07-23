use num_traits::Float;

use super::lag::lag;
use crate::data::{Instant, Scalar};
use crate::graph::Segment;
use crate::graph::cb::{Comp, Fork, Id, Right};
use crate::operators::num::{divide, subtract};
use crate::ports::ArrayPort;

/// `n`-tick relative change `(x − x₋ₙ) / x₋ₙ` (the YoY-growth shape):
/// `growth(n) @ x`. The subtraction is kept (rather than reduced to
/// `x / x₋ₙ − 1`-style rank equivalents) so a negative base keeps its faithful
/// sign. Self-recording; `NaN` until the lag is available. The one-tick
/// special case, without a private record, is
/// [`pct_change`](crate::operators::num::pct_change).
#[allow(clippy::type_complexity)] // the combinator tree is the documentation
pub fn growth<T: Scalar + Float, const N: usize>(
    n: usize,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Comp(
        Fork(Id::default(), lag(n)),
        Comp(Fork(subtract(), Right::default()), divide()),
    )
}
